# GraphRAG Chat

Streamlit-чат использует `graph_preprocess/C3_A4_posweight_model.pt` для
ранжирования узлов графа и локальную OpenAI-совместимую LLM для ответа.

Перед запуском загрузите в LM Studio модель
`qwen2.5-coder-14b-instruct` и включите Local Server на порту `1234`.

```powershell
docker compose up --build -d
```

После запуска приложение доступно по адресу <http://localhost:8501>.

Проверка состояния:

```powershell
docker compose ps
docker compose logs -f graph-chat
```

Остановка:

```powershell
docker compose down
```

Для другого OpenAI-совместимого сервера задайте переменные перед запуском:

```powershell
$env:LLM_BASE_URL = "http://host.docker.internal:1234/v1"
$env:LLM_MODEL = "qwen2.5-coder-14b-instruct"
docker compose up --build -d
```
