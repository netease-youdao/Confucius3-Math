#!/bin/bash

# 设置环境变量
export GRADIO_DEFAULT_CONCURRENCY_LIMIT=20
export PYTHONPATH=.:$PYTHONPATH

MODEL_PATH="netease-youdao/Confucius3-Math"
SERVED_MODEL_NAME="confucius3-math"
PORT="8199"
MAX_MODEL_LEN="32768"
TP=1

# 使用vllm启动模型服务
echo "正在启动vLLM模型服务..."
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name $SERVED_MODEL_NAME \
  --host 0.0.0.0 \
  --port $PORT \
  --dtype auto \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_MODEL_LEN \
  --enforce-eager \
  --trust-remote-code &

# 等待几秒让模型服务有时间启动
sleep 5

# 启动Web Demo服务
echo "正在启动Web Demo服务..."
python3 web/stream_demo.py
