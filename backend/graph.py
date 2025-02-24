
import os
import sqlite3
import configparser

from typing import Any
from models import ChatState
from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from vector_store import vector_store
from langchain.schema import SystemMessage
from logger import graph_logger as logger
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import TavilySearchResults

# Load configuration
config = configparser.ConfigParser()
config.read("../config/config.ini")

# Set API keys securely 
os.environ["GROQ_API_KEY"] = config["api_key"]["groq_api_key"]
os.environ["TAVILY_API_KEY"] = config["api_key"]["tavily_api_key"]

class Node(ABC):
    """Abstract base class for all nodes in the graph."""
    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        """Method to be implemented by subclasses to process inputs and return outputs."""
        pass


class RAGNode(Node):
    def __init__(self, k: int, state: ChatState):
        self.k = k
        self.state = state
        self.vector_store = vector_store

    def invoke(self) -> dict:
        file_path = self.state["file_path"]
        query = self.state["messages"][-1].content
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
            state: ChatState
        ):
        self.max_results = max_results
        self.search_depth = search_depth
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.include_images = include_images
        self.state = state
        self.web_search_tool = TavilySearchResults(
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

    def invoke(self) -> dict:
        """Retrieve context from web search."""
        query = self.state["messages"][-1].content
        logger.info(f"WebSearchNode: Performing web search for query: '{query}'")
        context = self.web_search_tool.run(query)[0]['content']
        logger.debug(f"WebSearchNode: Retrieved context: {context[:200]}...")
        return {"web_context": context}


class LLMNode(Node):
    def __init__(self, model_name: str, temperature: float, state: ChatState):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatGroq(name=model_name, temperature=temperature)
        self.state = state

    def invoke(self) -> dict:
        """Generate response using LLM while maintaining conversation history."""
        logger.info(f"LLMNode: Generating response using model '{self.model_name}' with temperature {self.temperature}")
        if self.state["rag_context"]:
            self.state["messages"].append(SystemMessage(f"RAG Context: {self.state['rag_context']}"))
        if self.state["web_context"]:
            self.state["messages"].append(SystemMessage(f"Web Context: {self.state['web_context']}"))
        logger.info(f'Messages - {self.state["messages"]}')
        response = self.llm.invoke(self.state["messages"])
        return {"messages": [response]}
    

def get_graph(
        k: int, 
        max_result: int, 
        search_depth: str, 
        include_answer: bool, 
        include_raw_content: bool, 
        include_images: bool, 
        model_name: str, 
        temperature: float, 
        sqlite_url: str
    ):
    logger.info("Initializing LangGraph Workflow...")
    graph = StateGraph(ChatState)
    logger.info("Adding nodes to the graph...")
    graph.add_node("rag_node", lambda state: RAGNode(k, state).invoke())
    graph.add_node("web_search_node", lambda state: WebSearchNode(max_result, search_depth, include_answer, include_raw_content, include_images, state).invoke())
    graph.add_node("llm_node", lambda state: LLMNode(model_name, temperature, state).invoke())

    def router(state: ChatState) -> list:
        enable_rag = True if state["file_path"] else False
        enable_web_search = state["enable_web_search"]
        if enable_rag and enable_web_search:
            logger.info("Router: Routing to ['rag_node', 'web_search_node']")
            return ["rag_node", "web_search_node"]
        elif enable_rag:
            logger.info("Router: Routing to ['rag_node']")
            return ["rag_node"]
        if enable_web_search:
            logger.info("Router: Routing to ['web_search_node']")
            return ["web_search_node"]
        else:
            return ["llm_node"]

    graph.add_conditional_edges(START, router)
    graph.add_edge("rag_node", "llm_node")
    graph.add_edge("web_search_node", "llm_node")
    graph.add_edge("llm_node", END)
    logger.info("Graph structure successfully defined.")
    with sqlite3.connect(sqlite_url, check_same_thread=False) as conn:
        memory = SqliteSaver(conn)
        compiled_graph = graph.compile(checkpointer=memory)
    logger.info("Graph compilation completed successfully.")
    return compiled_graph
