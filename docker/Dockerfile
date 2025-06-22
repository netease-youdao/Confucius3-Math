# 使用官方 Python 镜像作为基础镜像
ARG CUDA_VERSION=12.1.0
ARG from=nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04
FROM ${from} AS base

# 设置阿里云镜像源并安装Python
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install huggingface_hub[hf_xet] vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 暴露服务端口
EXPOSE 8827
EXPOSE 8199

# 启动服务
CMD ["sh", "run_service_stream.sh"]