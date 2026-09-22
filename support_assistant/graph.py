import os
import json
from typing import TypedDict, List
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

try:
    import anthropic
except ImportError:
    anthropic = None


PROMPT_TEMPLATE = """You are a helpful and precise customer support assistant for Zepto, a quick-commerce grocery delivery service.

CONTEXT (retrieved policy excerpts):
{context}

TASK: Answer the customer's question using ONLY the information in the CONTEXT above.

NEGATIVE CONSTRAINT: Do not answer using information not present in the provided context. If the context does not contain the answer, say so explicitly rather than guessing.

FEW-SHOT EXAMPLE:
Question: "What is your delivery fee?"
Context: "Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Answer: "Delivery is free on orders over INR 149. Orders below that incur a flat INR 25 delivery fee."

FORMAT: Respond with a single, direct, well-formed paragraph (2-4 sentences). Do not repeat the question. Do not include any preamble like "Based on the context".

LENGTH: Keep the answer under 80 words.

Customer question: {query}
Answer:"""



STRUCTURED_OUTPUT_SUFFIX = """

Respond with ONLY a single valid JSON object (no markdown fences, no preamble, no trailing text) matching exactly this schema:
{{"answer": "<your answer as a string, following the FORMAT and LENGTH rules above>", "sources": {sources}, "confidence": <float between 0.0 and 1.0>}}
"""

MAX_LLM_RETRIES = 3


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float


def call_real_llm(prompt: str) -> str:
    """
    Placeholder for the actual Groq (or other free-tier) LLM API call.
    Only used when MOCK_LLM=0. Not implemented for the graded mock baseline.
    """
    raise NotImplementedError("Real LLM call is only used in the optional MOCK_LLM=0 extension.")


def call_llm_with_retry(prompt: str, max_retries: int = 2) -> AnswerResponse:
    """
    Optional MOCK_LLM=0 path: calls the real LLM and validates its raw
    output against the AnswerResponse schema. If validation fails, retries
    up to `max_retries` additional times with a corrective instruction
    appended to the prompt, before giving up and returning a marked error.
    """
    current_prompt = prompt
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            raw_output = call_real_llm(current_prompt)
            return AnswerResponse.model_validate_json(raw_output)
        except ValidationError as e:
            last_error = e
            current_prompt = (
                prompt
                + f"\n\nYour previous response did not match the required JSON "
                  f"schema (fields: answer:str, sources:list[str], confidence:float). "
                  f"Error: {e}. Please respond again with ONLY valid JSON matching "
                  f"that schema."
            )



  
    return AnswerResponse(
        answer=f"[ERROR] LLM output failed schema validation after {max_retries + 1} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )





class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_ids: List[str]
    retrieved_texts: List[str]
    answer: str
    sources: List[str]
    confidence: float


POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_collection("zepto_policies")

# Only instantiate a real API client when the real-LLM path is actually
# selected, so mock-mode runs never require an API key or network access.
_llm_client = None
if anthropic is not None and os.environ.get("MOCK_LLM", "1") == "0":
    _llm_client = anthropic.Anthropic()


def _call_llm_raw(prompt: str) -> str:
    """One call to the real LLM. Lets transport/API errors propagate to the retry loop."""
    response = _llm_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_structured_response(raw: str) -> dict:
    """Strip optional markdown fences and parse the JSON body."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def call_llm_structured(base_prompt: str, sources: List[str], max_retries: int = MAX_LLM_RETRIES) -> AnswerResponse:
    """
    Calls the real LLM and validates its output against the AnswerResponse
    Pydantic schema. On a parse or validation failure, the error is fed back
    into the prompt so the model can self-correct, and the call is retried
    up to `max_retries` times. If every attempt fails (bad output on every
    retry, or a transport/API error), falls back to a safe canned response
    instead of raising, so the graph never crashes.
    """
    prompt = base_prompt + STRUCTURED_OUTPUT_SUFFIX.format(sources=json.dumps(sources))
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = _call_llm_raw(prompt)
            parsed = _parse_structured_response(raw)
            return AnswerResponse(**parsed)

        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            # Model produced output that didn't parse or didn't match the schema.
            # Feed the error back in and ask it to correct itself.
            last_error = e
            prompt = (
                base_prompt
                + STRUCTURED_OUTPUT_SUFFIX.format(sources=json.dumps(sources))
                + f"\n\nYour previous response was invalid ({e}). "
                  f"Return ONLY the corrected JSON object, nothing else."
            )

        except Exception as e:
            # Transport/API-level failure (network, auth, rate limit, etc.).
            # Retry as-is; no prompt content to fix.
            last_error = e

    # All retries exhausted -- degrade gracefully rather than raising.
    return AnswerResponse(
        answer="I'm unable to generate a reliable answer right now. Please try again shortly.",
        sources=sources,
        confidence=0.0,
    )


def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()
    if os.environ.get("MOCK_LLM", "1") != "0":
      
        intent = "policy_question" if any(kw in query_lower for kw in POLICY_KEYWORDS) else "general_question"
    else:
        intent = "policy_question" if any(kw in query_lower for kw in POLICY_KEYWORDS) else "general_question"
    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    query_emb = embed_model.encode([state["query"]]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    retrieved_ids = results["ids"][0]
    retrieved_texts = results["documents"][0]

    if os.environ.get("MOCK_LLM", "1") != "0":
        top_chunk_snippet = retrieved_texts[0][:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        context = "\n".join(retrieved_texts)
        prompt = PROMPT_TEMPLATE.format(context=context, query=state["query"])
        structured = call_llm_structured(prompt, sources=retrieved_ids)
        answer = structured.answer
        confidence = structured.confidence

    return {
        **state,
        "retrieved_ids": retrieved_ids,
        "retrieved_texts": retrieved_texts,
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": confidence,
    }


def direct_answer(state: GraphState) -> GraphState:
    if os.environ.get("MOCK_LLM", "1") != "0":
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = "[real-LLM direct answer path -- not used in mock mode]"

    return {
        **state,
        "retrieved_ids": [],
        "retrieved_texts": [],
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


def route_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


app_graph = build_graph()


def ask(query: str) -> AnswerResponse:
    initial_state: GraphState = {
        "query": query, "intent": "", "retrieved_ids": [], "retrieved_texts": [],
        "answer": "", "sources": [], "confidence": 0.0,
    }
    result = app_graph.invoke(initial_state)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: policy question (should trigger retrieval)")
    print("=" * 70)
    r1 = ask("What is your delivery fee?")
    print(r1.model_dump_json(indent=2))

    print("\n" + "=" * 70)
    print("Test 2: general question (should NOT trigger retrieval)")
    print("=" * 70)
    r2 = ask("What's the weather like today?")
    print(r2.model_dump_json(indent=2))