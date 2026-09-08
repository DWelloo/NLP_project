from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, Language, TextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.chunking_strategy import ChunkingStrategy


def get_splitter(strategy: ChunkingStrategy, **kwargs) -> TextSplitter:
    chunk_size = kwargs.get("chunk_size", 1000)
    chunk_overlap = kwargs.get("chunk_overlap", 200)
    
    if strategy.value == ChunkingStrategy.RECURSIVE.value:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators", ["\n\n", "\n", " ", ""])
        )

    elif strategy.value == ChunkingStrategy.FIXED.value:
        return CharacterTextSplitter(
            separator=kwargs.get("separator", "\n\n"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif strategy.value == ChunkingStrategy.SEMANTIC.value:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return SemanticChunker(embeddings)
    
    elif strategy.value == ChunkingStrategy.MARKDOWN.value:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    else:
        raise ValueError(f"Nieznana strategia chunkowania: {strategy}")