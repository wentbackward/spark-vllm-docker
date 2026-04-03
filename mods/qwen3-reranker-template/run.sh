#!/bin/bash
# Install Qwen3 reranker chat template
echo "Installing Qwen3 reranker chat template"
cp /workspace/mod-content/qwen3_reranker.jinja /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/chat_templates/qwen3_reranker.jinja
echo "=======> to apply, use --chat-template qwen3_reranker.jinja"
