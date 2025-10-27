import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = "docs"  # txt 파일들 위치
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 2


# -------------------------
# 1) 데이터 로딩 + 청크 분할
# -------------------------
def load_txt_documents(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",  # 하위 폴더까지 포함하려면 "**/*.txt"
        loader_cls=TextLoader,  # 개별 파일 로더 지정
        show_progress=True,  # 로드 진행상황 표시 (선택)
    )

    docs = loader.load()
    return docs


raw_docs = load_txt_documents(DATA_DIR)
print(raw_docs[0].page_content[:300])  # 앞부분 300자 출력

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)
all_splits = text_splitter.split_documents(raw_docs)

print(f"Chunked into {len(all_splits)} pieces")

# -------------------------
# 2) 임베딩
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------------
# 3) Chroma 벡터스토어 (자동 로컬 저장)
# -------------------------
CHROMA_DB_DIR = "chroma_db_wmd"

from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="wmd_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_wmd",  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(documents=all_splits)

# -------------------------
# 4) Retriever
# -------------------------

results = vector_store.similarity_search("낙타바위와의 사연을 알려줘", top_k=TOP_K)

print(results[0])
