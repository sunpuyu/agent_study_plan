from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# 全局设置，完整打印、单行
np.set_printoptions(threshold=np.inf, linewidth=9999, precision=6)
# 直接加载本地路径，不再访问镜像；基于脚本所在目录，避免受运行目录影响
# model_path = Path(__file__).resolve().parent / ".all-MiniLM-L6-v2"
model_path = Path(__file__).resolve().parent.parent.parent / ".bge-small-zh-v1.5"
model = SentenceTransformer(str(model_path))

s1 = "小猫在沙发上睡觉"
s2 = "猫咪躺在沙发休息"
s3 = "汽车在马路上行驶"

v1 = model.encode(s1)
v2 = model.encode(s2)
v3 = model.encode(s3)

import numpy as np
def cos_sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

print("向量shape：", v1.shape)
print("v1完整向量：", v1)
print("s1-s2 余弦相似度：", cos_sim(v1, v2))
print("s1-s3 余弦相似度：", cos_sim(v1, v3))