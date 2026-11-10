# 极简骨架，你可以直接扩充
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
import os, shutil
from openai import OpenAI
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class LocalBGEEmbedding(Embeddings):
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

envpath = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(envpath)

# 1. 准备文本（硬编码，不用PDF）
raw_texts = [
    "小猫在沙发上睡觉，它每天下午两点左右休息。",
    "猫咪喜欢吃冻干，不喜欢很烫的食物。",
    "汽车在马路上行驶，家用轿车限速一般60km/h。",
    "小狗喜欢出门散步，早上和晚上是遛狗时间。"
]

# 2. 加载Embedding（沿用你Day1的BGE）
model_path = Path(__file__).resolve().parent.parent.parent / ".bge-small-zh-v1.5"
embedding = LocalBGEEmbedding(model_path=str(model_path))

# 3. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents(raw_texts, metadatas=[{"source":"animal"},{"source":"animal"},{"source":"car"},{"source":"dog"}])

faiss_db_path = "./faiss_demo"
if os.path.exists(faiss_db_path):
    shutil.rmtree(faiss_db_path)

# 4. Chroma入库（持久化）
faiss_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)
faiss_db.save_local(faiss_db_path)

# 5. 检索：query，拿top3
new_faiss = FAISS.load_local(faiss_db_path, embedding, allow_dangerous_deserialization=True)
query = "小猫一般什么时候睡觉？"
res = new_faiss.similarity_search(
                query, 
                k=3, 
                filter={"source":"animal"})
print("=====召回片段=====")
for doc in res:
    print(doc.page_content, id(doc))

# 6. 拼接Prompt（RAG核心，把召回内容塞进去）
prompt_template = """基于下面的参考内容回答问题，不知道就说不知道，不要编造：
参考内容：
{context}
问题：{question}
回答：
"""
context = "\n".join([d.page_content for d in res])
final_prompt = prompt_template.format(context=context, question=query)
# print("\n=====给LLM的Prompt=====")
# print(final_prompt)

# 【这里接ollama/OpenAI兼容接口调用LLM】
# 读取环境变量 DEEPSEEK_API_KEY
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise RuntimeError("未读取到环境变量 DEEPSEEK_API_KEY，请先配置！")

client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-flash", 
    messages=[
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.1  # RAG场景温度调低，减少幻觉
)

# 输出LLM结果
answer = response.choices[0].message.content
print("\n=====DeepSeek 返回答案=====")
print(answer)