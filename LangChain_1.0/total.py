from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

DATA_DIR = "docs"  # txt 파일들 위치
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4

# -------------------------
# 1) 데이터 로딩 + 청크 분할
# -------------------------
def load_txt_documents(data_dir: str):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",           # 하위 폴더까지 포함하려면 "**/*.txt"
        loader_cls=TextLoader,     # 개별 파일 로더 지정
        show_progress=True         # 로드 진행상황 표시 (선택)
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
chunks = text_splitter.split_documents(raw_docs)

print(f"Chunked into {len(chunks)} pieces")