from src.exception import CustomException
from src import logger
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma


class DataLoader:
    """
    A class to handle data loading, processing, and storage for RAG (Retrieval-Augmented Generation).

    Methods:
        __init__(): Initializes the embedding model.
        fetch_contents(): Fetches data from specified URLs.
        store_data(): Stores processed data into a vector database.
        create_retriever(): Creates a retriever for similarity-based document search.
    """

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings()

    def fetch_contents(self):
        try:

            # List of URLs to load documents from
            urls = [
                "https://medium.com/@fraidoonomarzai99/introduction-to-generative-ai-and-llm-in-depth-aaf4bb5546ff",
                "https://medium.com/@fraidoonomarzai99/retrieval-augmented-generation-rag-in-depth-e90a05c38a02"
            ]

            # load the data
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"documents: {len(docs_list)}")
            return docs_list

        except Exception as e:
            raise CustomException(e, sys)

    def store_data(self):
        try:
            docs_list = self.fetch_contents()

            # split the data into chunks
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=300, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs_list)

            # add to vectorDB
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name="rag-chroma",
                embedding=self.embeddings,
            )
            logger.info("data stored")

            return vectorstore

        except Exception as e:
            raise CustomException(e, sys)

    def create_retriever(self):
        try:
            vectorstore = self.store_data()
            retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
            logger.info("create_retriever")

            return retriever

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     data_loader = DataLoader()
#     retriever = data_loader.create_retriever()
#     print(str(retriever.invoke("what is an rag")[0]))
