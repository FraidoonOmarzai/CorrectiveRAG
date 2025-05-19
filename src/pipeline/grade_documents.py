from src.exception import CustomException
from src import logger

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from src.pipeline.data_loader import DataLoader

import os
import sys
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.

    Attributes:
        binary_score (str): Indicates whether the document is relevant ('yes' or 'no').
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGrader:
    """
    Class to grade the relevance of retrieved documents to a user's question.
    Uses the ChatGroq model with structured output.

    Methods:
        __init__(): Initializes the model, retriever, and prompt.
        create_grader(): Creates and returns the grader and RAG chain.
    """

    def __init__(self):
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant")
        self.retriever = DataLoader().create_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")

    def create_grader(self):
        """
        Creates a grader and RAG chain to evaluate document relevance.

        Returns:
            tuple: (retrieval_grader, rag_chain)
                - retrieval_grader: A chain to evaluate document relevance.
                - rag_chain: A chain for retrieval-augmented generation.

        Raises:
            CustomException: If an error occurs during the creation of the grader or chain.
        """
        try:

            # LLM with function call
            structured_llm_grader = self.llm.with_structured_output(
                GradeDocuments)

            # Prompt
            system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
                If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

            grade_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human",
                     "Retrieved document: \n\n {document} \n\n User question: {question}"),
                ]
            )

            retrieval_grader = grade_prompt | structured_llm_grader
            logger.info("retrieval_grader")

            # question = "Generative AI"
            # docs = self.retriever.invoke(question)
            # doc_txt = docs[1].page_content
            # print(30*"==")
            # results = retrieval_grader.invoke({"question": question, "document": doc_txt})
            # print(results)
            # print(30*"==")

            # Chain
            rag_chain = self.prompt | self.llm | StrOutputParser()
            logger.info("rag_chain")

            # generate
            # generation = rag_chain.invoke({"context": docs, "question": question})

            # print(30*"==")
            # print(generation)
            # print(30*"==")

            return retrieval_grader, rag_chain
        except Exception as e:
            raise CustomException(e, sys)


# if __name__=="__main__":
#     retrieval_grade = RetrievalGrader()
#     retrieval_grade, rag_chain = retrieval_grade.create_grader()
