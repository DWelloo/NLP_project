from enum import Enum

class ChunkingStrategy(Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"