from dotenv import load_dotenv
from importlib.metadata import version
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from typing import List
import tempfile
import os
from lang_chain.loaders.document_loaders import load_pdf_file

load_dotenv()
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

def create_knowledgeBase():
    """ Cria uma vector store a partir de arquivos."""
    #CHUNK
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = load_pdf_file(os.getenv("PDF_PATH"))
    chunks = splitter.split_documents(documents)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# fase 1 indexing: LOAD -> CHUNCK -> EMBED -> STORE
# fase 2 quering: QUERY -> SEARCH -> RETRIEVE -> GENERATE/AUGMENT -> ANSWER
def main():
    vector_store = create_knowledgeBase()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = ChatOllama(model="llama3.2", temperature=0.3)

    #RAG PROMPT
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:
        {context}
        Question: {question}
        Answer:
        
        Make sure to answer in a concise manner, and if you don't know the answer, just say "I don't know."
        """
    )
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "Who was James Cook?",
        "To which country did Vasco da Gama want to travel?"
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        




if __name__ == "__main__":
    main()

