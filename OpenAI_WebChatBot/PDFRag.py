import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

pdf_dir = "./pdfs/"  # PDF 파일이 저장된 폴더 경로
documents = []
dirs = os.listdir(pdf_dir)

def combine_docs(path):
    loader = PyPDFLoader(path).load()

    temp_text = ""
    for temp in loader:
        temp_text += temp.page_content + "\n\n"

    return Document(page_content=temp_text)

# 1. PDF 폴더 내 모든 파일 불러오기
for dir in dirs:
    full_path = os.path.join(pdf_dir, dir)
    temp_doc = combine_docs(full_path)
    documents.append(temp_doc)

# 2. 텍스트 나누기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

# 3. 임베딩 생성 및 벡터 저장소 구축
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# 4. QA 체인 구성
retriever = db.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4.1-nano"),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

# 5. 질문 루프
while True:
    question = input("질문을 입력하세요 (종료하려면 'exit'): ")
    if question.lower() == "exit":
        break

    result = qa_chain.invoke({"query": question})
    print("💬 답변:", result["result"])
