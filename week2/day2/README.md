# Day2 完整学习路径（理论 30min + 动手 90min，按顺序来，可直接照着做）

前置：你 Day1 已经搞定了 `RecursiveCharacterTextSplitter`、BGE/miniLM embedding，环境已经装好 `langchain`、`sentence-transformers`
先装依赖：

```
uv add chromadb faiss-cpu langchain-community
```

> 
> faiss-cpu 适合本地学习；如果后面有 GPU 再换 faiss-gpu

## 一、理论 30min（拆成 3 块，每块 10min，边看边记笔记）

### 10min｜Chroma 核心

- Chroma 定位：**开箱即用、自带持久化、自带元数据、适合小 / 中型 RAG 原型**，学习首选
  - 持久化：指定`persist_directory="./chroma_db"`，重启程序向量还在；不指定就是内存版，程序退出丢失
  - metadata：给每一段 chunk 打标签（来源、序号、时间），后续可以**过滤检索**（比如只召回某一篇文档片段）
  - 检索方式：默认余弦相似度（和我们 Embedding 配套）
- 一句话笔记：Chroma = 向量存储 + 索引 + 元数据管理 + 相似度检索，封装很高，不用自己管索引

### 10min｜FAISS 核心

- FAISS 定位：Meta 出品，**纯向量索引库，没有元数据、没有持久化、没有文档绑定**，主打速度，适合百万、千万级向量
- 两种距离：
  - `IndexFlatL2`：欧氏距离，距离越小越相似
  - `IndexFlatIP`：内积，**归一化后的 embedding，内积等价余弦相似度**（我们 BGE 模型输出一般要归一化，这点很关键）
- 一句话笔记：FAISS 只存向量数组，你自己要维护【向量→原文 + 元数据】的映射字典，检索返回下标，你自己查表拿文本

### 10min｜向量检索流程 + 核心认知（重中之重，对应你笔记那句）

标准链路：
`用户query → query embedding → 在向量库中算相似度 → 取出top-k文本片段 → 拼接进Prompt → LLM生成回答`
✅ 必背结论：

> 
> 向量库**只会按向量距离召回片段，不理解语义**；召回的内容本身是错的、无关的，LLM 再强也容易顺着错误内容编造（幻觉）
> 额外理解：

- Top-k：召回多少条候选片段；k 太大，容易混入噪声；k 太小，信息不足
- 区分：**检索（召回） ≠ 重排（rerank）**，今天暂时不加重排，先最小闭环

> 
> 自测（学完理论问自己）：Chroma 和 FAISS 最大区别是什么？FAISS 为什么需要自己维护向量和原文的映射？

---

## 二、动手 90min（分 4 个阶段，循序渐进，不要一下子写全）

> 
> 目标：先 Chroma 最小 RAG 闭环跑通 → 看懂日志 → 改成 FAISS 版本对比 → 可选 MMR 实验

### 阶段 1：Chroma 最小 RAG Demo（40min，优先完成，拿到正向反馈）

脚本逻辑：硬编码几段短文 → 切分 chunk → embedding → chroma 入库 → 用户提问 → 检索 topk → 组装 prompt → 调用 LLM 问答
推荐用**本地轻量 LLM（比如 ollama 的 qwen:1.8b）**，如果你还没 ollama，可以先把【检索 + 拼接 prompt】单独跑，打印 prompt 看召回效果，先不调用 LLM。

```
# 极简骨架，你可以直接扩充
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 准备文本
raw_texts = [
    "小猫在沙发上睡觉，它每天下午两点左右休息。",
    "猫咪喜欢吃冻干，不喜欢很烫的食物。",
    "汽车在马路上行驶，家用轿车限速一般60km/h。",
    "小狗喜欢出门散步，早上和晚上是遛狗时间。"
]

# 2. 加载Embedding
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 3. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents(raw_texts)

# 4. Chroma入库持久化
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db_demo"
)
vector_db.persist()

# 5. 检索
query = "小猫一般什么时候睡觉？"
retriever = vector_db.as_retriever(search_kwargs={"k":3})
res = retriever.get_relevant_documents(query)
print("=====召回片段=====")
for doc in res:
    print(doc.page_content)

# 6. 拼接Prompt
prompt_template = """基于下面的参考内容回答问题，不知道就说不知道，不要编造：
参考内容：
{context}
问题：{question}
回答：
"""
context = "\n".join([d.page_content for d in res])
final_prompt = prompt_template.format(context=context, question=query)
print("\n=====给LLM的Prompt=====")
print(final_prompt)

```

✅ 本阶段验收：

1. 运行后能看到召回的片段，问小猫的问题，不会大量召回汽车那段
2. 改 query：`汽车限速多少`，观察召回变化
3. 改 k=1 /k=2，对比召回结果差异

### 阶段 2：理解元数据 + 过滤检索（20min）

改造上面代码，给 chunk 加 metadata：

```
chunks = splitter.create_documents(
    raw_texts,
    metadatas=[{"source":"animal"},{"source":"animal"},{"source":"car"},{"source":"animal"}]
)
```

检索时加过滤：只召回 source=animal 的内容

```
res = vector_db.similarity_search(query, k=3, filter={"source":"animal"})
```

理解：Chroma 可以做元数据过滤，FAISS 原生没有这个能力，需要自己实现。

### 阶段 3：FAISS 版本复刻（20min，对比差异）

> 
> 重点体会：FAISS 只存向量，你要自己维护 `index -> 原文` 的映射字典

```
from langchain_community.vectorstores import FAISS

# 前面的chunks、embedding复用
faiss_db = FAISS.from_documents(chunks, embedding)
# FAISS没有持久化，手动保存到本地
faiss_db.save_local("./faiss_demo")
# 加载
new_faiss = FAISS.load_local("./faiss_demo", embedding, allow_dangerous_deserialization=True)

# 检索
res_faiss = new_faiss.similarity_search(query, k=3)
print("FAISS召回结果：")
for d in res_faiss:
    print(d.page_content)
```

✅ 对比实验（写进笔记）：
同一个 query，Chroma 召回 vs FAISS 召回，结果是不是几乎一致？

> 
> 因为底层都是 BGE embedding + 余弦相似度；差异不是 “准不准”，而是功能、持久化、元数据支持。

### 阶段 4：可选拓展：MMR 检索（10min，理解多样性）

MMR（最大边际相关性）：在保证和 query 相似的前提下，**减少召回片段之间的重复冗余**

```
# Chroma开启MMR检索
retriever_mmr = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "fetch_k":5} # 先取5个候选，再挑3个不重复的
)
res_mmr = retriever_mmr.get_relevant_documents(query)
print("=====MMR召回=====")
for d in res_mmr:
    print(d.page_content)
```

对比普通相似度检索：

- 普通 similarity：优先取最像 query 的，内容容易重复
- MMR：兼顾相关性 + 多样性，适合文档内容高度重复的场景；代价是偶尔会召回相关性略低的片段

---

# 三、当日笔记模板（直接抄）

1. Chroma：高封装、自带持久化、元数据过滤，适合快速原型学习；默认余弦相似度
2. FAISS：底层向量索引，速度快，适合大规模；**无元数据、无持久化，需要自己维护文档映射**
3. 检索链路：query embedding → 相似度匹配 → top-k 召回 → 拼接上下文 → LLM 生成
4. 核心结论：向量库只负责召回，**召回错了 LLM 救不回来**，RAG 上限由检索质量决定
5. MMR：牺牲一点点相关性，降低召回片段的冗余，提升多样性

# 四、踩坑提前预警（你大概率会遇到）

1. FAISS 加载报错 `allow_dangerous_deserialization`：新版本 langchain 要求显式开启，上面代码已经加上
2. Chroma 持久化：不要手动乱改 chroma_db 文件夹，容易损坏索引
3. embedding 不统一：入库、检索必须**同一个 embedding 模型**，不然向量空间不一样，检索完全乱掉（高频坑！）
4. MMR 的 fetch_k：必须大于 k，fetch_k 是先召回的候选池大小

# 五、验收标准（今天结束前确认）

- ✅ Chroma 最小 RAG 闭环跑通，能输入问题、拿到 LLM 回答
- ✅ 会写元数据过滤检索
- ✅ FAISS 版本跑通，能对比两者召回结果
- ✅ 理解 MMR 和普通相似度检索的区别（可选）

如果你想要，我可以把【Chroma 完整版（带 ollama 调用）+ FAISS 版本 + MMR 对比】整合成一份可直接运行的完整 py 文件。