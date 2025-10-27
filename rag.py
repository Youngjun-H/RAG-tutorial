# from langchain_unstructured import UnstructuredLoader
# from unstructured.cleaners.core import clean_extra_whitespace

# loader = UnstructuredLoader(
#     "docs/docs_18.상징솔.txt",
#     post_processors=[clean_extra_whitespace],
# )

# docs = loader.load()

# docs[5:10]

# from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

import os

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


loader = DirectoryLoader(
    "docs",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

# 각 문서의 파일명을 메타데이터로 추가
for d in docs:
    d.metadata["source"] = d.metadata["source"].split("/")[-1]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0
)
splits = splitter.split_documents(docs)

# 3️⃣ OpenAI 임베딩 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 4️⃣ 벡터스토어 구축 및 저장
if not os.path.exists("vectorstore"):
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("vectorstore")
else:
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# 5️⃣ Retriever 구성
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

template = """
    당신은 주어진 짧은 문서들의 내용을 기반으로 질문에 답하는 AI 어시스턴트입니다.
    아래에 관련 문서 내용이 주어집니다. 문서의 주요 내용을 종합하여 질문에 대한 답변을 생성하세요.

    문서 내용:
    {context}

    질문: {question}

    답변:
"""
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
)