下面给你一套**就对着这 40 分钟理论 + 配套动手的学习方案**，分三块：Embedding 语义向量 & 余弦相似度、词袋 vs 语义匹配、RecursiveCharacterTextSplitter，告诉你看什么、怎么理解、怎么自测，不发散、不引入额外高级内容。

> 
> 前提：这 40min 是**理解概念**，后面还有动手代码验证；不要一上来啃论文。
> 工具：LangChain 文档 + SentenceTransformers 示例 + 计算器手算感受余弦相似度（可选），B 站只用来卡点补图。

## 一、Embedding、语义向量、余弦相似度（≈20min）

### 先搞懂一句话

Embedding：把不定长自然语言文本，映射成一个固定长度的浮点数向量，**语义相近的句子，向量在高维空间距离更近**。

> 
> 重点：它不是关键词编码，是语义的压缩表示。

### 学习顺序

1. 先看这个核心比喻（5min）
想象：每一句话是空间里一个点；意思差不多的句子挨在一起；意思完全无关的离很远。
余弦相似度：衡量两个向量**方向夹角**，不在乎向量长度。

> 
> 对比：L2 距离是空间直线距离；Chroma 默认用余弦相似度检索。
   - 余弦值越接近 1 → 方向几乎一样 → 语义相似
   - 接近 0 → 无关
   - 接近 -1 → 语义相反
2. 极简代码跑一眼向量长什么样（不用深入模型训练，只是感知）

```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

s1 = "小猫在沙发上睡觉"
s2 = "猫咪躺在沙发休息"
s3 = "汽车在马路上行驶"

v1 = model.encode(s1)
v2 = model.encode(s2)
v3 = model.encode(s3)
print(v1.shape) # 384维向量
```

你只需要观察：`s1`和`s2`语义接近，余弦相似度高；`s1`和`s3`很低。

> 
> 不需要自己实现余弦计算，知道它的含义即可。

3. 区分：词袋匹配（关键词匹配） VS 语义匹配（Embedding 匹配）【这块是重中之重】

- **词袋（关键词匹配，比如 BM25）**：只看有没有相同字 / 词，不认意思
例子：
> 
> 文档：苹果手机续航很强
> 查询：iPhone 电池耐用
> 词袋匹配：几乎没有重合词 → 匹配不到
> 查询：苹果手机屏幕大
> 词袋匹配：有 “苹果手机” → 命中，但语义不是你想要的
- **语义匹配（Embedding）**：看含义，不怕同义词、换说法、句式改写
> 
> “苹果手机续航很强” 和 “iPhone 电池耐用” embedding 相似度很高，可以召回。

✅ 自测问题（关上资料能不能答出来）

1. 余弦相似度 = 0.9 代表什么？0.1 呢？
2. 关键词匹配的优点和致命缺点是什么？
3. Embedding 解决了关键词匹配的什么问题？
4. Embedding 能不能 100% 准确理解语义？（答：不能，会有语义漂移，这也是后面幻觉、检索错误的根源）

> 
> 卡点辅助：B 站搜：`Embedding 余弦相似度 词袋和语义检索区别`，挑 10 分钟内的讲解，看懂就停。

---

## 二、RecursiveCharacterTextSplitter（≈20min）

### 核心知识点清单（你要掌握的全部）

1. **递归分割优先级（顺序从上到下）**

```
"\n\n" 段落分隔 → "\n" 换行 → " " 空格 → "" 字符
```

逻辑：优先尝试按段落切；如果段落长度依然超过 chunk_size，才降级用换行切；换行还太长，再空格；最后没办法只能硬拆字符。

> 
> 为什么叫递归？不断尝试分隔符，直到分片长度满足上限。
> ❗ 坑：它**不理解语义**，只看文本符号 + 字符长度，不是智能按句子语义边界切。

2. 两个参数定义

- `chunk_size`：单个分片最大字符数（注意：默认是**字符数，不是 token**！后面有 TokenTextSplitter 才是按 token）
- `chunk_overlap`：相邻两个分片之间重叠多少字符。

3. 重叠窗口的作用

> 
> 一句话：**防止一个完整语义单元，被一刀切成两半，分散在两个 chunk 里，检索时只拿到一半信息**
> 举例子：一段说明一共 700 字，chunk_size=500。
> 前一个 chunk：0~500；后一个 chunk：450~950（overlap=50），中间 50 个字重叠。
> 如果关键结论刚好落在 470-520 位置，两个分片都会带上一部分，检索时有机会召回完整信息。

4. overlap 的坑（必须记）

- overlap 太小：断语义，一句话被劈成两半，信息残缺，后面内源幻觉高发
- overlap 太大：大量重复内容入库，向量库冗余，检索的时候一次性召回很多重复片段，污染上下文，LLM 容易混淆

> 
> 经验关系：一般 overlap 设置为 chunk_size 的 10% ~ 20%。比如 chunk=500 → overlap=50~100。

### 学习方式

1. 打开 LangChain 官方文档页面：RecursiveCharacterTextSplitter，看参数说明和示例
https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/
2. 写极简测试脚本（就是你 Day1 要写的），打印切出来的分片，肉眼观察

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """在这里粘贴一大段连贯文字，比如一段PDF复制出来的内容
包含多个段落、换行。用来测试切分效果。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len # 按字符计数
)
chunks = splitter.split_text(text)
for i, c in enumerate(chunks):
    print(f"==== chunk {i}, len={len(c)} ====")
    print(c)
```

然后你改三组参数对比：

- (200, 20)
- (500, 50)
- (1000, 100)
肉眼看：哪里把一句话拦腰切断了、哪里重复太多。这就是你 Day1 的核心实验。

✅ 自测问题

1. RecursiveCharacterTextSplitter 的分割符优先级顺序是什么？
2. chunk_size 是硬上限吗？会不会偶尔超一点？
3. chunk_overlap 是干嘛的，是不是越大越好？
4. 它能不能识别一个完整逻辑段落，保证不切断语义？（答案：不能，它只是基于符号规则）

---

## 三、整套 40 分钟时间分配实操版（直接照着执行）

1. 0~12min：Embedding 向量概念 + 余弦相似度，跑一句 encode 看向量 shape，理解高维点
2. 12~20min：词袋匹配 vs 语义匹配，自己举 2 个正反例子，写在笔记里
3. 20~32min：RecursiveCharacterTextSplitter 分割规则、chunk_size/overlap 含义
4. 32~40min：梳理 overlap 的利弊，写下两条坑，在脑子里预判：chunk 太小会怎么样、overlap 太大怎么样

## 四、笔记模板（直接抄进你的学习文档）

```
1. Embedding
- 作用：文本→固定长度向量，语义相近向量靠近
- 相似度：余弦相似度 [-1,1]，越接近1越相似
- 词袋匹配：匹配关键词，无法理解同义改写；容易漏召回、误召回
- 语义匹配：Embedding，擅长同义、转述；缺点：存在语义漂移

2. RecursiveCharacterTextSplitter
- 分割优先级：双换行 → 换行 → 空格 → 字符
- chunk_size：分片最大长度（字符）
- chunk_overlap：分片重叠长度，防止语义被切断
- 经验：overlap≈chunk_size的10%~20%
- 局限：规则切分，不理解语义，依然可能切断逻辑句
```

## 五、避坑提醒（很多人这里踩）

1. 不要去学 transformer 预训练、对比学习损失函数，本周不需要知道模型怎么训练出来的；你只需要知道**输入文本，输出向量，向量可以算相似度**
2. 不要混淆字符长度和 token 长度：RecursiveCharacterTextSplitter 默认 len () 统计字符，不是 token；后面 LLM 输入限制才关心 token
3. 不要幻想分片可以完美：**不存在万能 chunk 参数，分片好坏依赖文档本身的行文风格**，这也是后面实验要验证的。

如果你想要，我可以给你准备一段测试长文本，专门用来测试切分（里面埋长句、跨段落结论，方便你直观看到 overlap 的价值）。