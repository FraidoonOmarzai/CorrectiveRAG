from src.exception import CustomException
from src import logger

from pprint import pprint
import sys

from src.pipeline.data_loader import DataLoader
from src.pipeline.grade_documents import RetrievalGrader
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START


from typing import List
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question (str): User input question.
        generation (str): Generated response.
        web_search (str): Indicates whether to perform web search.
        documents (List[str]): List of retrieved documents.
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


class CRAGWorkFlow:
    """
    Corrective Retrieval-Augmented Generation (CRAG) Workflow.

    This class implements a pipeline for dynamically generating accurate responses by combining
    retrieval, document grading, query transformation, and LLM-based generation.

    Methods:
        - __init__: Initializes the workflow components (LLM, Retriever, Grader, Web Search Tool).
        - retrieve: Retrieves documents based on a question.
        - generate: Generates an answer using the retrieved documents.
        - grade_documents: Filters relevant documents from the retrieved set.
        - transform_query: Reformulates the question for better search efficiency.
        - web_search: Searches the web for additional information if required.
        - decide_to_generate: Determines whether to generate a response or reformulate the query.
        - build_graph: Constructs and compiles the state graph for the workflow.
    """

    def __init__(self):

        self.llm = ChatGroq(model_name="llama-3.1-8b-instant")
        self.retriever = DataLoader().create_retriever()
        self.retrieval_grader, self.rag_chain = RetrievalGrader().create_grader()
        self.web_search_tool = TavilySearchResults(max_results=3)

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        try:

            logger.info("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = self.retriever.invoke(question)
            return {"documents": documents, "question": question}

        except Exception as e:
            raise CustomException(e, sys)

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        try:

            logger.info("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            # RAG generation
            generation = self.rag_chain.invoke(
                {"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}

        except Exception as e:
            raise CustomException(e, sys)

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        try:

            logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            # Score each doc
            filtered_docs = []
            web_search = "No"
            for d in documents:
                score = self.retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"
                    continue
            return {"documents": filtered_docs, "question": question, "web_search": web_search}

        except Exception as e:
            raise CustomException(e, sys)

    def question_rewriter(self):
        try:

            system = """You a question re-writer that converts an input question to a better version that is optimized \n 
                for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

            re_write_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                    ),
                ]
            )

            question_rewriter = re_write_prompt | self.llm | StrOutputParser()
            return question_rewriter

        except Exception as e:
            raise CustomException(e, sys)

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        try:

            logger.info("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]

            # Re-write question
            que_rewriter = self.question_rewriter()
            better_question = que_rewriter.invoke({"question": question})
            return {"documents": documents, "question": better_question}

        except Exception as e:
            raise CustomException(e, sys)

    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """
        try:

            logger.info("---WEB SEARCH---")
            question = state["question"]
            documents = state["documents"]

            # Web search
            docs = self.web_search_tool.invoke({"query": question})
            if all(isinstance(d, str) for d in docs):
                web_results = "\n".join(docs)
            else:
                web_results = "\n".join(
                    [d.get("content", str(d)) for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)

            return {"documents": documents, "question": question}

        except Exception as e:
            raise CustomException(e, sys)

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        try:

            logger.info("---ASSESS GRADED DOCUMENTS---")
            state["question"]
            web_search = state["web_search"]
            state["documents"]

            if web_search == "Yes":
                # All documents have been filtered check_relevance
                # We will re-generate a new query
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                print("---DECISION: GENERATE---")
                return "generate"

        except Exception as e:
            raise CustomException(e, sys)

    def build_graph(self):
        try:

            logger.info("build_graph...")
            workflow = StateGraph(GraphState)

            # Define the nodes
            workflow.add_node("retrieve", self.retrieve)  # retrieve
            # grade documents
            workflow.add_node("grade_documents", self.grade_documents)
            workflow.add_node("generate", self.generate)  # generatae
            # transform_query
            workflow.add_node("transform_query", self.transform_query)
            workflow.add_node("web_search_node", self.web_search)  # web search

            # Build graph
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "generate",
                },
            )
            workflow.add_edge("transform_query", "web_search_node")
            workflow.add_edge("web_search_node", "generate")
            workflow.add_edge("generate", END)

            # Compile
            return workflow.compile()

        except Exception as e:
            raise CustomException(e, sys)


# if __name__=="__main__":
#     workflow = CRAGWorkFlow()
#     app = workflow.build_graph()

#     # Run
#     inputs = {"question": "Explain generative ai?"}
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             # Node
#             pprint(f"Node '{key}':")
#         pprint("\n---\n")

#     # Final generation
#     pprint(value["generation"])
