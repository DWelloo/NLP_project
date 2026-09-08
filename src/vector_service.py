from langchain_text_splitters import TextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.chunking import get_splitter
from src.chunking_strategy import ChunkingStrategy
from src.file_loaders import get_loader
from src.logger import get_logger


class VectorService:
    def __init__(self, url: str, collection_name: str, embeddings: HuggingFaceEmbeddings):
        self._logger = get_logger("VectorService")
        self._url = url
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._qdrant = QdrantClient(url=self._url)
        self._vector_store = self._init_vector_store()


    def set_chunking_strategy(self, chunking_strategy: ChunkingStrategy, chunk_size: int = None, chunk_overlap: int = None):
        self.current_chunking_config: tuple[ChunkingStrategy, int, int] = (chunking_strategy, chunk_size, chunk_overlap)
        self._splitter: TextSplitter = get_splitter(
            strategy=chunking_strategy, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )


    def clear_db(self):
        self._logger.debug(f"Deleting collection {self._collection_name} from Qdrant...")
        if self._collection_name in [c.name for c in self._qdrant.get_collections().collections]:
            self._qdrant.delete_collection(collection_name=self._collection_name)
        self._init_vector_store()


    def ingest_document(self, file_path: str):
        self._logger.debug(f"Splitting document {file_path}...")
        loader = get_loader(file_path)
        docs = self._splitter.split_documents(loader.load())
        
        self._logger.debug(f"Insert {len(docs)} document chunks into Qdrant...")
        self._vector_store.add_documents(docs)


    def search(self, query: str, k: int = 3) -> list:
        result = self._vector_store.similarity_search(query, k=k)
        self._logger.debug(f"Vector search results: {result}")
        return result


    def _init_vector_store(self):
        if any(c.name == self._collection_name for c in self._qdrant.get_collections().collections):
            self._logger.debug(f"Connecting to existing Qdrant collection: {self._collection_name}")
            return QdrantVectorStore.from_existing_collection(
                embedding=self._embeddings,
                collection_name=self._collection_name,
                location=self._url
            )
        else:
            self._logger.debug(f"Creating new Qdrant collection: {self._collection_name}")
            return QdrantVectorStore.from_documents(
                [], self._embeddings, location=self._url, collection_name=self._collection_name
            )   