# NLP_projekt_7

### Starting Docker containers
```
docker-compose up
```

### Starting the application
1. Wait for the Docker containers (Neo4j, Qdrant, Ollama) to start.
2. Prepare a virtual environment:
```
python -m venv .venv
.venv/Scripts/activate
pip install uv
uv sync
```
3. Copy **.env.example** to **.env**. It is worth experimenting with **OLLAMA_CHAT_MODEL**, **EMBEDDING_MODEL**, and **OLLAMA_GRAPH_MODEL**. The remaining settings can be left at their default values (the database configuration must match the settings in **docker-compose.yml**).
4. The models configured in **.env** under **OLLAMA_CHAT_MODEL** and **OLLAMA_GRAPH_MODEL** must be downloaded. For example, to download **gemma:2b**:
```
docker exec -it ollama_llm ollama pull gemma:2b
```
5. Start the application:
```
uv run streamlit run .\main.py
```
6. The UI is available at [localhost:8501](http://localhost:8501).




