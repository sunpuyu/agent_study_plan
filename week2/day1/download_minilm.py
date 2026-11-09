import os
# 【重点】必须在import huggingface_hub之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

local_dir = snapshot_download("BAAI/bge-small-zh-v1.5", local_dir="./bge-small-zh-v1.5")
print("下载完成，模型路径：", local_dir)
