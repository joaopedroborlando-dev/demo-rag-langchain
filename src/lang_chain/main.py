from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

print(f"langchain_core version {core_version}")
print(f"langgraph version {lg_version}")

def main():
    llm = ChatOllama(model_name="llama3.2", temperature=0.3)
    response = llm.invoke("Say 'seetup complete!' in one word")
    print(f"Response from Ollama: {response}")


if __name__ == "__main__":
    main()

