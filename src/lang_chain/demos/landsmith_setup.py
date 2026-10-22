"""
LangSmith Setup e Observabilidade
Prod monitor para LangChain/LangGraph
"""

import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="rag_demo")
def demo_basic_tracing(topic):
    """
    LangSmith tracing demo
    """
    llm = ChatOllama(model="llama3.2", temperature=0.3)

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence."
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": topic})

    print(f"Result:{result}")
    print("\nCheck LangSmith dashboard for trace details")

def call_demo():
    demo_basic_tracing("Santos Dumont")
    demo_basic_tracing("Milton Santos")
    demo_basic_tracing("Machado de Assis")

if __name__ == "__main__":
    call_demo()