#!/bin/bash
# Fixed Qwen3.6 chat template from froggeric/Qwen-Fixed-Chat-Templates.
# Fixes the "No user query found in messages" exception that broke tool-calling
# when conversations ended on a <tool_response> turn. Also adds <|think_on|>/
# <|think_off|> system flags and `developer` role support.
# Source: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/blob/main/qwen3.6/chat_template.jinja
set -e
cp chat_template.jinja $WORKSPACE_DIR/qwen3.6.jinja
echo "=======> to apply chat template, use --chat-template qwen3.6.jinja"
