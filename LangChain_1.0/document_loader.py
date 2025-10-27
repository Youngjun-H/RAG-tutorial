from langchain_community.document_loaders import DirectoryLoader, TextLoader

# TXT 파일이 들어있는 디렉토리 경로 지정
DIRECTORY_PATH = "./docs"

# DirectoryLoader를 사용하여 모든 .txt 파일 로드
loader = DirectoryLoader(
    DIRECTORY_PATH,
    glob="**/*.txt",           # 하위 폴더까지 포함하려면 "**/*.txt"
    loader_cls=TextLoader,     # 개별 파일 로더 지정
    show_progress=True         # 로드 진행상황 표시 (선택)
)

# 모든 문서를 한 번에 불러오기
documents = loader.load()

# 확인
print(f"총 {len(documents)}개의 문서를 불러왔습니다.")
print("첫 번째 문서 내용 미리보기:\n")
print(documents[0].page_content[:300])  # 앞부분 300자 출력