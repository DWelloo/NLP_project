from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.document_loaders.base import BaseLoader


def get_loader(file_path: str) -> BaseLoader:
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        return TextLoader(file_path, encoding="utf8")