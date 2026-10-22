import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
import tempfile

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

document = '''
Captain James Cook: A Brief Biography
Introduction Captain James Cook (1728-1779) was a British explorer, navigator, cartographer, and captain in the Royal Navy. He is best known for his three groundbreaking voyages to the Pacific Ocean, during which he achieved the first recorded European contact with the eastern coastline of Australia and the Hawaiian Islands, as well as the first recorded circumnavigation of New Zealand.
Early Life and Career Born on October 27, 1728, in Marton, Yorkshire, Cook started his maritime career in the merchant navy as a teenager. In 1755, he enlisted in the Royal Navy. He demonstrated exceptional skill in cartography and mathematics during the Seven Years' War, surveying the entrance to the Saint Lawrence River, which allowed British forces to capture Quebec. His meticulous charts caught the attention of the Admiralty and the Royal Society.

Um modelo de linguagem (como Llama, GPT ou Gemini) isolado atua apenas com base no próprio treinamento estático.
O LangChain funciona como a camada de orquestração que conecta esses modelos a fontes de dados privadas, APIs e ferramentas externas, padronizando o desenvolvimento em quatro frentes principais: PadronizaçãoInterface única para trocar entre OpenAI, Gemini, Anthropic ou Ollama alterando apenas uma linha.
RAG (Dados Próprios) Leitura de PDFs, divisão em blocos (chunks), geração de embeddings e busca em bancos vetoriais. LCEL (Chains).
Composição declarativa de fluxos usando o operador pipe ( | ) para encadear etapas.
Agentes e Tools Permite que a IA acione funções Python, consulte bancos SQL ou consuma APIs externas sob demanda.

Watercolor paint is a water-based artistic medium made of finely ground color pigments suspended in a water-soluble binder, which is typically natural gum arabic. When water is added, the paint dissolves and can be applied with a brush—usually onto paper—to create transparent, luminous washes.
Core Components
• Pigment: Fine natural or synthetic colored powders that provide the hue.
• Binder: Gum arabic (a sticky sap from the acacia tree) that holds the pigment particles together and adheres them to the paper once the water dries.
• Additives: Glycerin or honey/sugar syrups are often added to retain moisture, improve flexibility, and make dried paint easy to rewet.
• Solvent:
'''

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n','\n','. ',' ']
)

recursive_chunks = recursive_splitter.split_text(document)

# print(f'Recursive chunks: {len(recursive_chunks)}')
# for i, chunk in enumerate(recursive_chunks):
#     print(f'\\n--- Chunk {i+1} ({len(chunk)} chars) ---')
#     print(chunk[:100] + '...' if len(chunk) > 100 else chunk)

semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(document)
# print(f'Recursive chunks: {len(semantic_chunks)}')
# for i, chunk in enumerate(semantic_chunks):
#     print(f'\\n--- Chunk {i+1} ({len(chunk)} chars) ---')
#     print(chunk[:100] + '...' if len(chunk) > 100 else chunk)

diretory = tempfile.mkdtemp(),

recursive_vertorstore = Chroma.from_texts(
    texts=recursive_chunks,
    embedding=embeddings,
    collection_name='recursive_chunks',
    persist_directory=tempfile.mkdtemp(),
)

semantic_vertorstore = Chroma.from_texts(
    texts=semantic_chunks,
    embedding=embeddings,
    collection_name='semantic_chunks',
    persist_directory=tempfile.mkdtemp(),
)

test_queries = [
    'Whos was James Cook?',
    'What is watercolor?',
    'What is LangChain for?'
]

def test_retrieval(query,vectorstore,name):
    results = vectorstore.similarity_search(query, k=1)
    print(f'\\n{name} - Query: \"{query}\"')
    print(f'Retrieved: {results[0].page_content[:150]}')
    return results[0].page_content

print(f'\n{'='*60}')
print(' TESTE')
print(f'\n{'='*60}')

for query in test_queries:
    print(f'\n{'='*60}')
    recursive_result= test_retrieval(query,recursive_vertorstore, 'RECURSIVE')
    semantic_result= test_retrieval(query,semantic_vertorstore, 'SEMANTIC')