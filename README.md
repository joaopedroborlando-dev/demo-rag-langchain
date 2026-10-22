## Etapas do RAG:
indexing: LOAD -> CHUNCK -> EMBED -> STORE
quering: QUERY -> SEARCH -> RETRIEVE -> GENERATE/AUGMENT -> ANSWER

## Causas de falha
1.Bad Chunking -> ovelap + critérios no chunck
2.Embedding Mismatch 
3.Retrievel Noise -> hybrid search
4.Context Overflow
5.Hallucination