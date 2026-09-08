import os
from dotenv import load_dotenv

from src.logger import get_logger

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")
DATA_PATH = os.getenv("DATA_PATH")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
OPENAI_KEY = os.getenv("OPENAI_KEY")


logger = get_logger("config.py")
logger.debug(
    f"""Loaded config: 
        - QDRANT_URL={QDRANT_URL}
        - NEO4J_URI={NEO4J_URI}
        - NEO4J_USER={NEO4J_USER}
        - DATA_PATH={DATA_PATH}
        - EMBEDDINGS_MODEL={EMBEDDINGS_MODEL}
        - VECTOR_COLLECTION_NAME={VECTOR_COLLECTION_NAME}
        - GEMINI_API_KEY={'SET' if GEMINI_API_KEY else 'NOT SET'}
        - GEMINI_MODEL_NAME={GEMINI_MODEL_NAME}
        """
)
