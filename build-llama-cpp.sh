# Ensure the repo is checked out
git submodule update --init --recursive

# Build the repo
cd llama.cpp
cmake -B build
cmake --build build --config Release -j$(nproc)

# Copy the llama-cli tool to the parent directory
cp build/bin/llama-cli ../llama-cli

# Return to the parent directory
cd ..
