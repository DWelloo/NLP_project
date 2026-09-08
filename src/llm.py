import google.generativeai as generativeai
import google.genai as genai

from src.logger import get_logger
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel


class BaseLlm:
    def generate(self, prompt: str, filepath: str = None) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_base_llm(self) -> BaseLanguageModel:
        raise NotImplementedError("This method should be overridden by subclasses.")


class GeminiLlm(BaseLlm):
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.1):
        self._logger = get_logger(__name__)
        generativeai.configure(api_key=api_key)
        self._model_name = model_name
        self._client = genai.client.Client(api_key=api_key)
        self._cypher_llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)

    def generate(self, prompt: str, filepath: str = None) -> str:
        try:
            contents = [prompt]
            if filepath:
                file = self._client.files.upload(file=filepath)
                self._logger.debug(f"Uploaded file {filepath} to Gemini API.")
                contents.insert(0, file)
            
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents
            )
            self._logger.debug(f"Gemini API response: {response.text}\n")
            return response.text if response.text else ""
        except Exception as e:
            self._logger.error(f"Error during Gemini API call: {str(e)}")
            raise e
        
    def get_base_llm(self) -> BaseLanguageModel:
        return self._cypher_llm