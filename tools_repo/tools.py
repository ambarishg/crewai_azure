from crewai import Agent, Task, Crew, LLM
from crewai_tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import SerperDevTool

from rag.rag_system import RAGSystem
from azureopenaimanager.azureopenai_helper import AzureOpenAIManager

@tool('SerperDevTool')
def search_serper(search_query: str):
    """Search the web for information on a given topic"""
    tool = SerperDevTool(
    search_url="https://google.serper.dev/search",
    n_results=2,
)

    return(tool.run(search_query=search_query))

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

@tool('CalculatorAdd')
def calculator_add(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Args:
        a (float): The first number to add.
        b (float): The second number to add.

    Returns:
        float: The sum of the two numbers.

    Raises:
        ValueError: If either of the inputs is not a number (int or float).
    """
    # Ensure inputs are valid numbers
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both inputs must be numbers (int or float).")
    
    # Perform addition
    return a + b

@tool('RAG_QDRANT')
def rag_qdrant(search_query:str):
    """Gets the answer from Qdrant"""
    rag_helper = RAGSystem(AzureOpenAIManager())
    return (rag_helper.query(search_query))


