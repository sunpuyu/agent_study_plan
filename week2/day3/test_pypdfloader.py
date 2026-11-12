from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../pdf_demo/B.pdf")
pages = loader.load()

print(f"一共读取到 {len(pages)} 页")
for idx, page in enumerate(pages):
    print(f"\n====第{idx+1}页，metadata：{page.metadata}====")
    print(page.page_content[:1000]) # 打印前1000字预览
