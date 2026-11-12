# 极简骨架，你可以直接扩充
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader  # 替代PyPDFLoader
from langchain_community.vectorstores import Chroma
from pathlib import Path
import os, shutil
from openai import OpenAI
from dotenv import load_dotenv

envpath = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(envpath)

# ========== 基线参数（固定，后面所有实验统一）==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
# ======================================================

loader = PyMuPDFLoader("../pdf_demo/A.pdf")
pages = loader.load()

# 2. 加载Embedding（沿用你Day1的BGE）
model_path = Path(__file__).resolve().parent.parent.parent / ".bge-small-zh-v1.5"
embedding = HuggingFaceEmbeddings(model_name=str(model_path))

# 3. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(pages)
print(f"从 {len(pages)} 页PDF, 一共切出来 {len(chunks)} 个chunk")


db_path = "./chroma_db_demo"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# 4. Chroma入库（持久化）
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=db_path
)
vector_db.persist()

# 5. 检索：query，拿top3
# query = "手机RTK算法的定位精度是多少？"
query = "手机的定位精度是多少？"
# query = "手机RTK的工作流程及各流程的耗时情况？"
# retriever = vector_db.as_retriever(search_kwargs={"k":TOP_K, score_threshold=0.7})
# res = retriever.invoke(query)

# 带分数检索
docs_with_score = vector_db.similarity_search_with_score(query, k=3)
for doc, score in docs_with_score:
    print(f"相似度：{score:.4f} | 页码：{doc.metadata['page']+1} | 内容摘要：{doc.page_content[:200]}")

# 组装上下文


# 6. 拼接Prompt（RAG核心，把召回内容塞进去）
prompt_template = """基于下面的参考内容回答问题，不知道就说不知道，不要编造：
参考内容：
{context}
问题：{question}
回答：
"""
context = "\n".join([f"【相似度{score:.3f}，页码{doc.metadata['page']+1}】{doc.page_content}" for doc, score in docs_with_score])
final_prompt = prompt_template.format(context=context, question=query)

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