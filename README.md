# gguf-graph-logger
A utility to log the structure of a GGML GGUF graph after it is constructed. This can be used to analyze the tensor sizes into and out of different nodes in the compute graph. It includes my fork of llama.cpp which adds the graph logging functionality.

## Pre-requisites
Make sure you've checked out the parent repo, [llama.cpp](https://github.com/LukeRouleau/llama.cpp). If you're running local LLMs, it is because of this codebase. Ollama, KoboldCpp, etc. are all based off of ggml and llama.cpp. All hail [ggerganov](https://github.com/ggerganov).

## Usage
### Build `llama.cpp`
1a. Run the build script in this repo. It simply makes sure that the `llama.cpp` submodule is checked out and runs the most generic build sequence in that repo.
```shell
./build-llama-cpp.sh
``` 

1b. If you want to manually build the `llama.cpp` repo yourself, or need to debug why this repo's build script did not work for you, check out the [build instruction](./llama.cpp/docs/build.md) from the repo. However you build, ensure you copy the `llama-cli` executable into the root of this repo. This repo's little build script does this for you, so refer to [it](./build-llama-cpp.sh) if you need more info. 

2. After building successfully, the `llama-cli` tool will be located in `./llama.cpp/build/bin/llama-cli`. This is where the python scripts expect it to be.

### Pick the HuggingFace GGUF models for which you want to log the graphs
1. Select the models for which you want to log the graph structure. All available HF GGUF models are browseable [here](https://huggingface.co/models?library=gguf&sort=trending).

2. Place the model names that you want to log the structure of in the list in [hf-gguf-models.json](./inputs/hf-gguf-models.json), which python script takes as input:
```json
[
    "TheBloke/Llama-2-7B-Chat-GGUF",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
]
```

### Run the python script [log-graphs.py](./log-graphs.py)
```python
python log-graphs.py
```

### Find the outputs in the output directory. There are three different types of output:
1. CSV - The raw tabular form of the graph structure.
2. ONNX - An ONNX-like protobuf openable in an ONNX viewer like [Netron](https://netron.app/), allowing for structure visualization.
3. IMG - PNG plots of some data analysis results like model operation composition, model size, etc.