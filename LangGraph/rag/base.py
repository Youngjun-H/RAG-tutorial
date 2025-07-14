from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, List, Optional

from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 10
        self.vectorstore: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.chain: Optional[Any] = None

    @abstractmethod
    def load_documents(self, source_uris: str) -> List[Document]:
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs: List[Document], text_splitter) -> List[Document]:
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        """임베딩 모델을 생성합니다. 하위 클래스에서 오버라이드 가능합니다."""
        return OpenAIEmbeddings(model="text-embedding-3-small")

    def create_vectorstore(self, split_docs: List[Document]):
        """벡터스토어를 생성합니다. 하위 클래스에서 오버라이드 가능합니다."""
        return FAISS.from_documents(
            documents=split_docs, embedding=self.create_embedding()
        )

    def create_retriever(self, vectorstore):
        """검색을 수행하는 retriever를 생성합니다. 하위 클래스에서 오버라이드 가능합니다."""
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_model(self):
        """LLM 모델을 생성합니다. 하위 클래스에서 오버라이드 가능합니다."""
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def create_prompt(self):
        """프롬프트를 생성합니다. 하위 클래스에서 오버라이드 가능합니다."""
        return hub.pull("teddynote/rag-prompt-chat-history")

    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self):
        """RAG 체인을 생성합니다. Template Method 패턴의 핵심 메서드입니다."""
        if not self.source_uri:
            raise ValueError("source_uri가 설정되지 않았습니다.")

        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        model = self.create_model()
        prompt = self.create_prompt()

        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return self

    def query(self, question: str, chat_history: Optional[List] = None) -> str:
        """질문에 대한 답변을 생성합니다."""
        if not self.chain:
            self.create_chain()

        if chat_history is None:
            chat_history = []

        # 검색을 통해 관련 문서를 가져옵니다
        docs = self.retriever.get_relevant_documents(question)
        context = self.format_docs([doc.page_content for doc in docs])

        # 체인을 실행합니다
        return self.chain.invoke(
            {"question": question, "context": context, "chat_history": chat_history}
        )
