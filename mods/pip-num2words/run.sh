#!/bin/bash
# SmolVLM's HF image processor needs num2words (number-to-words text
# expansion). Not in the base vllm-node-tf5 image — install at launch.
# Referenced by the smolvlm2-*.yaml recipes' `mods:` block.
set -e
pip install --no-cache-dir num2words
echo "=======> num2words installed"
