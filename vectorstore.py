import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

EMBEDDING_DIM = 3072

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

index = faiss.IndexFlatL2(EMBEDDING_DIM)

print(faiss.__version__)