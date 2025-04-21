#!/usr/bin/env python3

import json
import os
import subprocess
import datetime
import sys
from pathlib import Path
import csv
import onnx
from onnx import helper, TensorProto, ValueInfoProto

JSON_PATH = "inputs/hf-gguf-models.json"
OUTPUT_DIR = "./outputs"

class Stage:
    def __init__(self, stage_name: str, width: int = 120):
        self.stage_name = stage_name
        self.width = width
        
    def __enter__(self):
        print(f"\n{'>' * 7} STARTING: {self.stage_name} {'>' * (self.width - len(self.stage_name) - 18)}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{'<' * 7} FINISHED: {self.stage_name} {'<' * (self.width - len(self.stage_name) - 18)}\n")


def get_models() -> list[dict]:
    with Stage("Loading models"):
        with open(JSON_PATH, 'r') as f:
            models = json.load(f)
        return models


def create_output_dirs() -> tuple[Path, Path, Path]:
    with Stage("Creating output directories"):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(OUTPUT_DIR) / timestamp
        
        csv_dir = output_dir / "csv"
        onnx_dir = output_dir / "onnx"
        img_dir = output_dir / "img"
        
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        return csv_dir, onnx_dir, img_dir


def log_graph_to_csv(model_name: str, csv_dir: Path, extra_args: str) -> Path | None:
    model_flat_name = model_name.replace('/', '_')
    output_file = csv_dir / f"{model_flat_name}.csv"
    
    cmd = f"./llama-cli -hf {model_name} {extra_args}"
    
    env = os.environ.copy()
    env["GGML_LOG_GRAPH"] = "1"
    env["GGML_LOG_GRAPH_FILENAME"] = output_file
    
    try:
        with Stage(f"Processing {model_name} to CSV"):
            subprocess.run(
                cmd,
                shell=True,
                env=env,
                check=True
            )
        print(f"Successfully processed {model_name} to CSV\n\n")
    except subprocess.CalledProcessError as e:
            print(f"Error processing {model_name}: {e}")
            print(f"Continuing with next model...")
            return None
    return output_file


def convert_to_onnx(csv_file: Path, onnx_dir: Path) -> Path | None:
    with Stage(f"Converting {csv_file} to ONNX"):
        nodes_data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            # Clean up node_id by removing '0x' prefix for consistency if present
            for row in reader:
                 if row['node_id'].startswith('0x'):
                     row['node_id'] = row['node_id'][2:]
                 for i in range(10): # Clean up src IDs as well
                    src_key = f'src{i}'
                    if src_key in row and row[src_key] != '(nil)':
                        if row[src_key].startswith('0x'):
                            row[src_key] = row[src_key][2:]
                    else:
                        row[src_key] = None # Use None for missing sources
                 nodes_data.append(row)

        if not nodes_data:
            print(f"No data found in {csv_file}")
            return None

        model_name_stem = csv_file.stem
        node_id_to_data = {node['node_id']: node for node in nodes_data}

        # --- Refined ONNX Conversion ---

        computational_nodes_data = []
        computational_node_ids = set()
        parameter_node_ids = set() # Track IDs of PARAM/LEAF nodes

        # Separate computational nodes from parameters/leaves
        for node_data in nodes_data:
            # Assume nodes marked PARAM or LEAF are weights/parameters, not compute steps
            # Also skip nodes with op 'NONE' if they exist
            if node_data['flags'] in ('PARAM', 'LEAF') or node_data['op'] == 'NONE':
                 parameter_node_ids.add(node_data['node_id'])
            # Skip specific ops if they represent constants or placeholders unlikely to be compute
            elif node_data['op'] in ('CONST', 'RESHAPE'): # Example: Add ops to skip if needed
                 parameter_node_ids.add(node_data['node_id']) # Treat these like parameters/inputs for now
            else:
                 computational_nodes_data.append(node_data)
                 computational_node_ids.add(node_data['node_id'])

        all_tensor_infos = {} # Map node_id -> ValueInfoProto
        graph_inputs_vi = []  # List of ValueInfoProto for graph inputs
        onnx_nodes = []       # List of ONNX nodes

        # Create ValueInfoProto for all tensors (parameters, inputs, intermediates)
        for node_id, node_data in node_id_to_data.items():
            tensor_name = f"tensor_{node_id}"
            try:
                # Use float dimensions, default to 1 if missing/invalid
                dims = [
                    int(float(node_data.get('dim0', 1) or 1)),
                    int(float(node_data.get('dim1', 1) or 1)),
                    int(float(node_data.get('dim2', 1) or 1)),
                    int(float(node_data.get('dim3', 1) or 1))
                ]
                # Handle potential zero dimensions if necessary, replace with 1
                dims = [d if d > 0 else 1 for d in dims]

                vi = helper.make_tensor_value_info(
                    tensor_name,
                    TensorProto.FLOAT, # Assuming FLOAT, might need refinement based on actual types
                    dims
                )
                all_tensor_infos[node_id] = vi
            except (ValueError, TypeError) as e:
                print(f"Warning: Error creating ValueInfo for node {node_data.get('name', node_id)} ({node_id}): {e}. Using default shape [1,1,1,1].")
                vi = helper.make_tensor_value_info(
                    tensor_name, TensorProto.FLOAT, [1, 1, 1, 1]
                )
                all_tensor_infos[node_id] = vi

        # Identify graph inputs and create ONNX nodes
        processed_src_ids = set() # Track sources that are outputs of other compute nodes
        for node_data in computational_nodes_data:
            node_id = node_data['node_id']
            op_type = node_data['op']
            node_name = f"{node_data['name']}_{node_id}" # Make node name unique
            output_tensor_name = f"tensor_{node_id}"
            input_tensor_names = []

            for i in range(10):
                src_id = node_data.get(f'src{i}')
                if src_id and src_id != '(nil)': # Check src_id validity again
                    input_tensor_name = f"tensor_{src_id}"
                    input_tensor_names.append(input_tensor_name)

                    # If this src_id hasn't been seen as an input before AND
                    # it's not the output of another computational node,
                    # then it must be a graph input (parameter or actual input).
                    if src_id not in computational_node_ids and src_id not in processed_src_ids:
                        if src_id in all_tensor_infos:
                             graph_inputs_vi.append(all_tensor_infos[src_id])
                             processed_src_ids.add(src_id)
                        else:
                             # This case should ideally not happen if the CSV is complete
                             print(f"Warning: Graph input source ID {src_id} for node {node_name} not found in node data. Skipping input.")


            # Create the ONNX node
            onnx_node = helper.make_node(
                op_type=op_type,
                inputs=input_tensor_names,
                outputs=[output_tensor_name],
                name=node_name
                # Attributes like shape are now implicitly defined by the ValueInfoProto
                # doc_string=f"node_id: {node_id}" # Optional: add original ID for reference
            )
            onnx_nodes.append(onnx_node)

        # Determine graph outputs
        # Simple approach: Last node in the computational list is the output
        # More robust: Nodes whose output tensor is not used as input by any other node
        output_node_ids = set(computational_node_ids)
        for node_data in computational_nodes_data:
             for i in range(10):
                 src_id = node_data.get(f'src{i}')
                 if src_id and src_id in output_node_ids:
                     output_node_ids.remove(src_id)

        graph_outputs_vi = []
        if not output_node_ids and computational_nodes_data:
             # Fallback: use the last computational node if the above logic fails
             last_comp_node_id = computational_nodes_data[-1]['node_id']
             print(f"Warning: Could not determine graph output automatically. Using last computational node: {last_comp_node_id}")
             if last_comp_node_id in all_tensor_infos:
                 graph_outputs_vi.append(all_tensor_infos[last_comp_node_id])
        else:
             for output_id in output_node_ids:
                 if output_id in all_tensor_infos:
                     graph_outputs_vi.append(all_tensor_infos[output_id])

        if not graph_outputs_vi and computational_nodes_data:
             # Final fallback if still no outputs and there are nodes
             last_node_id = computational_nodes_data[-1]['node_id']
             if last_node_id in all_tensor_infos:
                  print(f"Warning: Final fallback for output: using last computational node {last_node_id}")
                  graph_outputs_vi.append(all_tensor_infos[last_node_id])


        # Create the graph
        if not onnx_nodes:
            print(f"Warning: No computational nodes found for {csv_file}. Skipping ONNX creation.")
            return None
        if not graph_inputs_vi:
             print(f"Warning: No graph inputs identified for {csv_file}.")
             # Optionally create a dummy input if needed by ONNX standard? For now, proceed.
        if not graph_outputs_vi:
             print(f"Warning: No graph outputs identified for {csv_file}. Cannot create valid ONNX graph.")
             return None


        graph_def = helper.make_graph(
            nodes=onnx_nodes,
            name=f"GGML_{model_name_stem}",
            inputs=graph_inputs_vi,  # Graph inputs (ValueInfoProto)
            outputs=graph_outputs_vi, # Graph outputs (ValueInfoProto)
            value_info=list(all_tensor_infos.values()) # All intermediate tensors' ValueInfoProto
        )

        # Create the model
        # Ensure the domain "GGML" used previously is handled or removed if sticking to standard ONNX ops
        # If custom ops are needed, define the domain: opset_imports=[helper.make_opsetid("GGML", 1)]
        model_def = helper.make_model(graph_def, producer_name="gguf-graph-logger")
        model_def.opset_import[0].version = 16 # Use a relevant ONNX opset version

        # Save the model
        output_file = onnx_dir / f"{model_name_stem}.onnx"
        onnx.save(model_def, str(output_file))

        print(f"Successfully converted to ONNX: {output_file}")
        return output_file


def main() -> None:
    models = get_models()
    assert len(models) > 0, "No models found in the JSON file."
    
    csv_dir, onnx_dir, img_dir = create_output_dirs()

    # Process each model
    print(f"Processing {len(models)} models to CSV")
    csv_files = []
    for model in models:
        model_name = model['name']
        extra_args = model.get('extra-cli-args', '')
        
        if output_file := log_graph_to_csv(model_name, csv_dir, extra_args):
            csv_files.append(output_file)
    print(f"All models processed to CSV in {csv_dir}")
    

    # Post-process each CSV file into an ONNX representation and plots
    for csv_file in csv_files:
        # Convert to ONNX
        onnx_file = convert_to_onnx(csv_file, onnx_dir)
        
        # Create plots
        # create_plots(csv_file, img_dir, onnx_file)
    

if __name__ == "__main__":
    main()
