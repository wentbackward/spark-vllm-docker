#!/bin/bash
# froggeric Qwen-Fixed-Chat-Templates, root chat_template.jinja (v22).
# Covers Qwen 3.5 / 3.6 / 3.8 from a SINGLE template — froggeric publishes
# one root file for all of them (there is no per-version qwen3.8/ dir).
#
# Why not the stock Qwen3.8 template: it still raises "No user query found
# in messages", which breaks tool-calling when a conversation ends on a
# <tool_response> turn — and it dropped `developer` role and the
# <|think_on|>/<|think_off|> flags. Verified 2026-08-15 on
# Qwen/Qwen3.8-27B-FP8: stock has 1 occurrence of the raise, 0 developer,
# 0 think_on; this file has 0 / 2 / 6.
#
# Source: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
set -e
cp chat_template.jinja $WORKSPACE_DIR/qwen3.8.jinja
echo "=======> to apply chat template, use --chat-template qwen3.8.jinja"
