from fastapi import FastAPI
from pydantic import BaseModel
from graph import ask, AnswerResponse

app = FastAPI(title="Zepto Support Assistant")


class QueryRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: QueryRequest) -> AnswerResponse:
    return ask(request.query)


@app.get("/")
def root():
    return {"status": "Zepto Support Assistant is running. Use POST /ask with a query field."}