# NLP_projekt_7

### Uruchomienie kontenerów Dockera
```
docker-compose up
```

### Uruchomienie aplikacji
1. Poczekać aż wstaną kontenery Dockera (Neo4j, Qdrant, Ollama)
2. Przygotować środowisko wirtualne:
```
python -m venv .venv
.venv/Scripts/activate
pip install uv
uv sync
```
3. Skopiować **.env.example** do **.env**. Warto eksperymentować z **OLLAMA_CHAT_MODEL**, **EMBEDDING_MODEL**, **OLLAMA_GRAPH_MODEL**. Resztę można zostawić domyślne (configi do baz danych muszą być zgodne z tymi ustawionymi w kontenerach Dockera w **docker-compose.yml**)
4. Modele, które ustawiliśmy w **.env** w **OLLAMA_CHAT_MODEL** i **OLLAMA_GRAPH_MODEL** muszą być pobrane, np. aby pobrać **gemma:2b**: 
```
docker exec -it ollama_llm ollama pull gemma:2b
```
5. Uruchomić aplikację:
```
uv run streamlit run .\main.py
```
6. UI dostępny na [localhost:8501](http://localhost:8501)




