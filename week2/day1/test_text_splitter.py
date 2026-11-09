from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """一、Embedding 究竟是什么？
在深入探讨技术细节之前，我们先打个比方来理解 Embedding的概念。想象一下，人类语言是一座巨大的图书馆，里面的每一本书（每一个文本）都蕴含着丰富的知识。但计算机就像一个不懂人类语言的外星人，它无法直接读懂这些书的内容。这时，Embedding 就像是一本神奇的翻译手册，它能将图书馆里的每一本书（文本）转化为计算机能理解的数字语言——向量。这样，计算机就能通过处理这些向量，来理解文本所表达的含义。
从技术角度来说，Embedding 是一种将高维稀疏的数据（如文本、图像、音频等）映射到低维稠密向量空间的技术。以文本为例，计算机无法直接对文字进行数学运算，而 Embedding 技术可以将文本中的每个词（token）转化为一个固定长度的向量，这些向量包含了词的语义信息。比如 “苹果” 这个词，经过 Embedding 后会得到一个向量，这个向量与 “水果”“香蕉” 等相关词的向量在向量空间中距离较近，而与 “汽车”“房子” 等不相关词的向量距离较远。通过这种方式，计算机就能根据向量之间的距离来判断词与词之间的语义关系。
二、Embedding 的工作流程详解
（一）数据预处理
在进行 Embedding 之前，首先需要对原始数据进行预处理。以文本数据为例，这一步通常包括去除特殊字符、转换为统一的大小写形式、分词等操作。比如对于句子 “Hello, world! I love programming.”，经过分词后可能会得到 [“Hello”, “world”, “I”, “love”, “programming”] 这样的 token 序列。
（二）构建词表
接下来，需要构建一个词表（vocabulary）。词表是所有可能出现的 token 的集合，每个 token 在词表中都有一个唯一的索引。例如，在上述例子中，“Hello” 可能对应索引 0，“world” 对应索引 1，以此类推。这个索引就像是 token 在向量空间中的 “地址”。
（三）生成 Embedding 向量
有了词表和索引后，就可以通过 Embedding 层来生成向量了。Embedding 层本质上是一个查找表（lookup table），它的行数等于词表的大小，列数就是我们所说的 Embedding 维度。当一个 token 的索引输入到 Embedding 层时，它会从查找表中找到对应的行，从而输出一个固定维度的向量。比如，当索引 0（即 “Hello”）输入时，Embedding 层会输出一个如 [0.1, 0.2, -0.3, 0.4] 这样的向量（这里只是示例，实际的向量值是通过训练学习得到的）。"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50,
    length_function=len # 按字符计数
)
chunks = splitter.split_text(text)
for i, c in enumerate(chunks):
    print(f"==== chunk {i}, len={len(c)} ====")
    print(c)
