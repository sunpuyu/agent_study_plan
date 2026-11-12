# Day3｜PDF RAG 完整学习路线（理论 20min + 动手 100min）

前置：Day2 的 Chroma 最小 RAG 脚本能正常跑、BGE embedding 环境就绪、已经装了 `langchain-community`
先装 PDF 依赖：

```
uv add pypdf
```

> 
> 说明：PyPDFLoader 底层就是 pypdf，只解析可复制文字的原生 PDF；**扫描件 PDF（图片型 PDF）它直接读不出文字**，这个是今天第一个重点边界。

## 一、理论 20min（分 2 块，边看边记笔记）

### 10min：PyPDFLoader 原理 & 局限

1. 原理：逐页读取 PDF，提取页面里的文字，一页输出一个 Document 对象，page_content 是该页文本，metadata 自带`page`页码、source 文件名。
2. 短板（一定要记，后面踩坑都会遇到）
   - ✅ 适合：文字版 PDF、普通论文、说明书、纯文字文档
   - ❌ 不适合：扫描 PDF、图片 PDF、手写 PDF、复杂表格、跨页公式、分栏排版（两栏论文经常文字乱序）
   - ❌ 没有排版理解：不会识别标题、段落，只是按页面扒文字；一页很长的话，整页文本一次性丢给切分器
3. 对比 Day2：Day2 是手动写文本；Day3 只是**数据源换成 Loader**，后面切分、Embedding、入库、检索、Prompt、LLM 这一整条链路完全复用，不需要重写。

### 10min：基线版本的意义（非常关键）

> 
> 今天的核心产出：**基线脚本，后面 Day4 幻觉测试、Day5 调参全部基于这份代码修改**
> 固定基线参数（全程统一，方便对照实验）

- chunk_size = 500
- chunk_overlap = 50
- top_k = 3
- 向量库：Chroma（优先，方便持久化）
- Embedding：BAAI/bge-small-zh-v1.5

> 
> 为什么固定：如果你每次实验参数都变，分不清是 chunk 影响效果，还是 top-k 影响，变量无法控制。

自测问题（学完理论自查）

1. PyPDFLoader 拿到的 Document 和 Day2 create_documents 出来的 Document 是不是同一个结构？（是的，所以后面链路可以无缝复用）
2. 扫描 PDF 为什么 PyPDFLoader 读不到内容？

---

## 二、动手 100min，分成 5 个阶段，顺序不要跳

> 
> 准备材料：准备 2 个 PDF
> PDF A：干净文档（比如普通说明文档、单栏文字 PDF，文字连续，排版整齐）
> PDF B：杂乱 PDF（两栏论文、文字挤在一起、段落被分页打断的文档）

### 阶段 1：单独跑 PyPDFLoader，先看提取效果（20min，先不接入 RAG）

目标：先验证能不能读出文字，观察分页带来的问题，不要直接堆进 RAG。
示例代码：

```
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./test_clean.pdf")
pages = loader.load()

print(f"一共读取到 {len(pages)} 页")
for idx, page in enumerate(pages):
    print(f"\n====第{idx+1}页，metadata：{page.metadata}====")
    print(page.page_content[:300]) # 打印前300字预览
```

✅ 实验观察点：

1. 看 metadata：自动带上 `source` 文件路径、`page` 页码，检索召回后可以告诉用户【答案来自第几页】，这是硬编码文本没有的能力
2. 看 PDF B（杂乱）：会不会出现文字乱跳、两栏 PDF 先读完左栏再右栏、一句话被页面截断
3. 记录：哪些页面提取出来的文本是碎的、不通顺的

> 
> 坑点：PyPDFLoader 默认按**页**切分 Document，一页可能几千字，所以必须交给 RecursiveCharacterTextSplitter 二次切 chunk！

### 阶段 2：迁移 Day2 RAG 代码，搭建基线 PDF RAG 脚本（40min，核心阶段）

链路：`PyPDFLoader加载PDF → 文本切分（固定500/50） → embedding → Chroma入库 → 检索topk=3 → 拼接prompt → LLM问答`

> 
> 直接复用 Day2 的检索、prompt、ollama 调用代码，只替换前面数据源部分
> 完整基线骨架：

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# ========== 基线参数（固定，后面所有实验统一）==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
# ======================================================

# 1. 加载PDF
loader = PyPDFLoader("./test_clean.pdf")
pages = loader.load()

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(pages)
print(f"一共切出来 {len(chunks)} 个chunk")

# 3. Embedding
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 4. 入库（持久化，每次测试可以改一下文件夹名避免混库）
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_baseline_pdf"
)
vector_db.persist()
retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})

# 5. 提问+召回
query = "文档里的xxx是什么？"
docs = retriever.get_relevant_documents(query)
context = "\n".join([f"【页码：{d.metadata['page']+1}】{d.page_content}" for d in docs])

prompt_template = """基于下面参考内容回答问题，不知道就说不知道，不要编造：
参考内容：
{context}
问题：{question}
回答：
"""
final_prompt = prompt_template.format(context=context, question=query)
print(final_prompt)

# 这里接上你Day2的ollama调用代码
```

✅ 本阶段验收：

1. 干净 PDF A 可以正常问答，召回片段附带页码
2. 保存这份脚本，命名 `rag_baseline.py`，**不删、不随便改这组参数**，后续所有调优都复制一份再修改
3. 笔记：split_documents 会继承 page 的 metadata，每个 chunk 都自带来源页码，这个非常有用

### 阶段 3：对比干净 PDF A 和杂乱 PDF B 的分片 & 问答效果（20min）

操作：

1. 把脚本里 PDF 路径换成杂乱 PDF B，**参数完全不动（500/50/k=3）**
2. 同样几个问题，对比两次实验：
   - 提取文字是否通顺？
   - chunk 会不会把一句话拦腰切断？
   - 相同问题，召回的内容质量、回答质量差别大不大？
   📌 笔记记录：

- 干净 PDF：chunk 语义连贯，召回稳定
- 分栏 / 分页打断的 PDF：原生提取缺陷，就算切分参数不变，RAG 效果天然变差，这是 Loader 层面的瓶颈，不是向量库的问题

### 阶段 4：基础测试 + 沉淀基线（20min）

1. 设计 3 类问题在基线脚本上测试：
   - ✅ 文档里明确写了的问题（应该答对）
   - ⚠️ 文档里没有的问题（理想情况回答 “不知道”，提前感受幻觉，为 Day4 铺垫）
   - 📌 需要跨 2 个 chunk 拼接信息才能回答的问题（看 top3 能不能同时召回相关片段）
2. 记录基线的现象：哪些问题答得好、哪些问题直接翻车
3. 仓库管理建议：
   - `rag_baseline.py` 固定基线版本
   - 后面 Day4 幻觉实验：复制一份 `rag_illusion_test.py`
   - Day5 调参实验：复制一份 `rag_tune_topk.py`

> 
> 好处：随时可以切回基线对比，不会改乱原始版本

---

## 三、当日笔记模板（直接写）

1. PyPDFLoader：只提取原生文字 PDF，输出带 page 元数据的 Document；扫描 PDF、分栏 PDF、表格提取能力弱
2. 链路迁移思路：RAG 是模块化的，**只替换加载器，后面切分、向量化、检索、LLM 链路完全复用**
3. 基线固定参数：chunk_size=500，overlap=50，top_k=3，作为后续所有实验对照基准
4. 每个 chunk 会继承原页面的 metadata，召回后可以输出页码，提升可信度
5. 上游文档提取质量，会直接决定 RAG 上限；Loader 的缺陷，靠后面调向量库很难修复

## 四、高频踩坑清单

1. Chroma 库混用：不同 PDF 入库不要用同一个 persist 目录，向量混在一起，检索错乱
2. 页码：PyPDFLoader 里 page 是从 0 开始，展示给用户记得 +1
3. 长文本截断：不要依赖 Loader，Loader 是按页加载，必须二次切分
4. 乱码：部分 PDF 编码特殊，会出现奇怪符号，属于 PDF 本身导出问题
5. 不要今天就改 chunk_size/top_k：今天目标是建立基线，调参留给 Day5

## 五、Day3 验收标准

✅ 跑通 PDF 版本 RAG，脚本命名基线版，参数固定
✅ 对比了干净 PDF、杂乱 PDF 两份文档的问答差异
✅ 召回结果可以输出来源页码
✅ 清楚 PyPDFLoader 的能力边界，知道扫描 PDF 它处理不了
✅ 基线脚本单独留存，不覆盖，留给后面两天实验

如果你想要，我可以直接给你一份**带 ollama 调用、打印召回日志、输出来源页码**的完整可直接运行的 `rag_baseline.py`。