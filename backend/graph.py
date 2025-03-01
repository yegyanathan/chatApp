
import os
import sqlite3
import configparser

from langchain_core.prompts import PromptTemplate
from typing import Any
from models import ChatState, Grade
from abc import ABC, abstractmethod
from vector_store import vector_store
from langchain.schema import SystemMessage
from logger import graph_logger as logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import TavilySearchResults
from langchain.chat_models import init_chat_model

# Load configuration
config = configparser.ConfigParser()
config.read("../config/config.ini")

# Set API keys securely 
os.environ["GROQ_API_KEY"] = config["api_key"]["groq_api_key"]
os.environ["TAVILY_API_KEY"] = config["api_key"]["tavily_api_key"]

class Node(ABC):
    """Abstract base class for all nodes in the graph."""
    @abstractmethod
    def invoke(self, state: ChatState) -> Any:
        """Method to be implemented by subclasses to process inputs and return outputs."""
        pass


class RAGNode(Node):
    def __init__(self, k: int):
        self.k = k
        self.vector_store = vector_store

    def invoke(self, state: ChatState) -> dict:
        file_path, query = state["file_path"], state["messages"][-1].content
        logger.info(f"RAGNode: Searching for query: '{query}' in file: '{file_path}' with k={self.k}")
        context = self.vector_store.similarity_search(query, k=self.k, expr=f'source == "{file_path}"')
        context = "\n".join([doc.page_content for doc in context])
        logger.debug(f"RAGNode: Retrieved context: {context[:200]}...")
        return {"rag_context": context}
    

class WebSearchNode(Node):
    def __init__(
            self, 
            max_results: int, 
            search_depth: str, 
            include_answer: bool, 
            include_raw_content: bool, 
            include_images: bool, 
        ):
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.web_search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

    def invoke(self, state: ChatState) -> dict:
        """Retrieve context from web search."""
        query = state["messages"][-1].content
        logger.info(f"WebSearchNode: Performing web search for query: '{query}'")
        context = self.web_search_tool.run(query)[0]['content']
        logger.debug(f"WebSearchNode: Retrieved context: {context[:200]}...")
        return {"web_context": context}


class LLMNode(Node):
    def __init__(self, model_name: str, temperature: float, model_provider):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = init_chat_model(model_name, temperature=temperature, model_provider=model_provider)

    def invoke(self, state: ChatState) -> dict:
        """Generate response using LLM while maintaining conversation history."""
        logger.info(f"LLMNode: Generating response using model '{self.model_name}' with temperature {self.temperature}")

        if state["rag_context"]:
            state["messages"].append(SystemMessage(f"RAG Context: {state['rag_context']}"))
        if state["web_context"]:
            state["messages"].append(SystemMessage(f"Web Context: {state['web_context']}"))

        # Append an instruction to highlight RAG-retrieved content
        state["messages"].append(SystemMessage(
            "Highlight any information taken from RAG or Web contexts using **...** in your response."
        ))
        logger.info(f'Messages - {state["messages"]}')
        response = self.llm.invoke(state["messages"])
        token_usage = response.response_metadata["token_usage"]
        logger.info('Token usage: %s', token_usage)
        return {"messages": [response]}
    

class FallbackNode(Node):
    def invoke(self, state: ChatState) -> dict:
        logger.info(f"FallbackNode: fallback response due to no relevant context found.")
        response = "I'm unable to find relevant information for your query right now. However, you can try rephrasing your question or providing more details, and I’ll do my best to assist you!"
        return {"messages": [response]}

    
class RelevanceChecker:
    def __init__(self, model_name: str, temperature: float, model_provider: str):
        self.model_name = model_name
        self.temperature = temperature
        self.model_provider = model_provider

        llm = init_chat_model(model_name, temperature=temperature, model_provider=model_provider)
        structured_llm = llm.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user query. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user query: {query} \n
            If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the query.""",
            input_variables=["context", "query"],
        )
        self.chain = prompt | structured_llm

    def invoke(self, query: str, context: str):
        scored_result = self.chain.invoke({"query": query, "context": context})
        score = scored_result.binary_score
        if score == "yes":
            logger.info("---DECISION: CONTEXT RELEVANT---")
            return True
        else:
            logger.info("---DECISION: CONTEXT NOT RELEVANT---")
            return False


def get_graph(
        k: int, 
        max_result: int, 
        search_depth: str, 
        include_answer: bool, 
        include_raw_content: bool, 
        include_images: bool, 
        model_name: str, 
        model_provider: str,
        temperature: float, 
        sqlite_url: str
    ):
    logger.info("Initializing LangGraph Workflow...")
    graph = StateGraph(ChatState)
    logger.info("Adding nodes to the graph...")
    graph.add_node("rag_node", lambda state: RAGNode(k).invoke(state))
    graph.add_node("web_search_node", lambda state: WebSearchNode(max_result, search_depth, include_answer, include_raw_content, include_images).invoke(state))
    graph.add_node("llm_node", lambda state: LLMNode(model_name, temperature, model_provider).invoke(state))
    graph.add_node("fallback_node", lambda state: FallbackNode().invoke(state))


    graph.add_edge(START, "rag_node")
    graph.add_conditional_edges("rag_node", lambda state: RelevanceChecker(model_name, temperature, model_provider).invoke(state["messages"][-1].content, state["rag_context"]), {True: "llm_node", False: "web_search_node"})
    graph.add_conditional_edges("web_search_node", lambda state: RelevanceChecker(model_name, temperature, model_provider).invoke(state["messages"][-1].content, state["web_context"]), {True: "llm_node", False: "fallback_node"})
    graph.add_edge("llm_node", END)
    logger.info("Graph structure successfully defined.")
    with sqlite3.connect(sqlite_url, check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        compiled_graph = graph.compile(checkpointer=memory)
    logger.info("Graph compilation completed successfully.")
    return compiled_graph
