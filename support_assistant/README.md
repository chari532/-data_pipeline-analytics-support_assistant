# Support Assistant Module (`/support_assistant`)

## What this does

A small, complete GenAI RAG (Retrieval-Augmented Generation) service for Zepto:
an 8-document policy corpus, embedded locally and indexed in ChromaDB, served
through a LangGraph-orchestrated flow that routes each query, retrieves
grounded context when needed, and returns a schema-validated JSON answer via
a FastAPI endpoint.

**LLM calls are fully mocked by default (MOCK_LLM unset or `1`)** -- the graded
baseline requires no signup, no API key, and no network call to any LLM
provider. A real LLM call (`MOCK_LLM=0`, via Groq's free tier) and a live
Hugging Face Spaces deployment are both optional, ungraded extensions.

## Install

```bash
pip install -r requirements.txt
```

Requires: `sentence-transformers`, `chromadb`, `langgraph`, `fastapi`, `uvicorn`, `pydantic`

For the optional `MOCK_LLM=0` extension only, also requires: `groq` (and a
`GROQ_API_KEY` environment variable). Neither is needed to run the graded
baseline.

## How to run (in order)

```bash
python ingest.py          # embeds the 8 policy docs and stores them in ChromaDB
uvicorn main:app --reload  # starts the FastAPI server on http://127.0.0.1:8000
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or POST
to `/ask` directly with a JSON body like `{"query": "..."}`.

Run with Docker instead:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Architecture: the RAG pipeline, stage by stage

**1. Ingestion** (`ingest.py`) -- Loads the 8 policy documents from `docs/*.txt`.
Each document is short enough to be treated as a single chunk (a simple
per-document chunking scheme), so no further splitting is applied. Every
chunk is tagged with its source document ID (e.g. `doc_01`) as metadata.

**2. Embedding** (`ingest.py`, using `sentence-transformers`) -- Each chunk's
text is encoded into a vector with the `all-MiniLM-L6-v2` model, which runs
entirely locally (no API key, no network call after the model is first
downloaded and cached).

**3. Storage** -- The embeddings, original text, and metadata are stored in a
ChromaDB `PersistentClient` collection named `zepto_policies` (persisted to
disk under `chroma_db/`), so `ingest.py` only needs to run once per machine.

**4. Retrieval** (`graph.py`, inside the `retrieve_and_answer` node) -- When a
query is classified as a `policy_question`, the query itself is embedded with
the same `all-MiniLM-L6-v2` model and used to query the ChromaDB collection
for the top-3 most similar chunks via cosine similarity. This retrieval step
runs for real in both MOCK_LLM modes, since it needs no API key.

**5. Generation** (`graph.py`) -- This is the only stage that branches on
`MOCK_LLM`:
- **Default (mock) state**: no LLM is called anywhere. `classify_intent` uses
  a keyword heuristic; `retrieve_and_answer` returns a canned string built
  from the top retrieved chunk (`f"Based on the retrieved context: {snippet}"`);
  `direct_answer` returns a fixed string for non-policy questions. The
  `AnswerResponse` Pydantic schema (`answer`, `sources`, `confidence`) is
  populated directly from code -- there is no LLM output to validate.
- **Optional `MOCK_LLM=0` state**: `classify_intent` and `direct_answer` would
  call a real LLM instead of the heuristic/fixed string. `retrieve_and_answer`
  does call a real LLM (via Groq's free tier) -- it prompts the model with the
  structured template in `graph.py` (`PROMPT_TEMPLATE`, following a
  role-context-task-format-length skeleton, with a negative constraint and a
  few-shot example), appends a JSON-schema instruction, and grounds the answer
  in only the retrieved chunks. The raw output is parsed and validated against
  the `AnswerResponse` Pydantic schema in `call_llm_structured()`; if parsing
  or validation fails, the error is fed back into the prompt and the call is
  retried, up to 2 additional times (3 attempts total). If every attempt still
  fails to validate -- or the API call itself fails (network, auth, rate
  limit) -- the function returns a clearly marked error response (the `answer`
  field is prefixed `[ERROR] ...` and `confidence` is `0.0`) instead of
  raising, so the graph never crashes.

**Routing**: `main.py` (FastAPI) receives the request and calls `graph.py`'s
`ask()` function, which runs the compiled LangGraph `StateGraph`. The graph
has a `classify_intent` entry node, then a conditional edge (`route_intent`)
that sends `policy_question` queries to `retrieve_and_answer` and
`general_question` queries to `direct_answer`; both terminate at `END`. This
routing logic is independent of `MOCK_LLM` -- only the generation step inside
each node branches on it.

## Data flow summary

```
docs/*.txt --(ingest.py: chunk + embed)--> ChromaDB "zepto_policies"

Client
  --> POST /ask {"query": "..."}          (main.py, FastAPI)
  --> ask(query)                          (graph.py)
  --> StateGraph.invoke(initial_state)
        --> classify_intent               (keyword heuristic, mock)
              --> route_intent (conditional edge)
                    --> policy_question   --> retrieve_and_answer
                                               (embed query -> ChromaDB top-3
                                                -> canned/LLM answer -> sources)
                    --> general_question  --> direct_answer
                                               (fixed string / LLM, no retrieval)
        --> END
  <-- AnswerResponse {answer, sources, confidence}   (validated Pydantic model)
  <-- JSON response
```

Each request flows through exactly one branch of the conditional edge --
`policy_question` queries pick up retrieved context and cite `sources`;
`general_question` queries skip retrieval entirely and return an empty
`sources` list. Both branches converge on the same `AnswerResponse` schema
before FastAPI serializes it back to the client.

## Example API calls

Both examples below were run with `MOCK_LLM` left at its default (unset --
the graded baseline), via `uvicorn main:app --reload` in one terminal and
the following in a second terminal (PowerShell, using `Invoke-RestMethod`
since `curl` is aliased to `Invoke-WebRequest` on Windows PowerShell):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -ContentType "application/json" -Body '{"query": "What is your delivery fee?"}' | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST -ContentType "application/json" -Body '{"query": "Whats the weather like today?"}' | ConvertTo-Json
```

(Equivalent on macOS/Linux: `curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"query": "..."}'`)

**Example 1 -- policy question (triggers retrieval)**

Request: `{"query": "What is your delivery fee?"}`

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_03"
  ],
  "confidence": 1.0
}
```

`classify_intent` matched the keyword "delivery", so the query was routed to
`retrieve_and_answer`: the top-3 most similar chunks (`doc_01`, `doc_05`,
`doc_03`) were retrieved from ChromaDB and are listed in `sources`.

**Example 2 -- general question (does not trigger retrieval)**

Request: `{"query": "Whats the weather like today?"}`

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

No policy keyword matched, so `classify_intent` routed the query to
`direct_answer` instead -- no embedding, no ChromaDB call, and an empty
`sources` list.