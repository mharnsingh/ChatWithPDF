from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from vectorstore import InitVectorStore
from rag_utils import CitedAnswer, format_citation, format_docs, \
    qa_prompt_template, agent_prompt_template
from schema import AgentRequest, AddDocsRequest

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from uuid import uuid4
import uvicorn
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "chat-with-pdf-demo"


# initialize rag components
k_retriever = 8
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
structured_llm = llm.with_structured_output(CitedAnswer)
vectorstore = InitVectorStore(retrieval_mode="hybrid")
retriever = vectorstore.as_retriever(search_kwargs={"k": k_retriever})


# initialize agent
@tool(response_format="content_and_artifact")
def retrieve_and_answer(query: str):
    """Retrieve information related to a query and generate an answer."""

    retrieved_docs = retriever.invoke(query)
    formatted_docs = format_docs(retrieved_docs)
    
    prompt = qa_prompt_template().invoke({"question": query, "context": formatted_docs})
    response = structured_llm.invoke(prompt)
    format_citations = format_citation(retrieved_docs, response.citations)
    answer = f"{response.answer}"
    if format_citations:
        answer += f"\n\nCitations:\n{format_citations}" 

    return answer, retrieved_docs

checkpointer = InMemorySaver()
agent = create_react_agent(
    llm, 
    tools=[retrieve_and_answer],
    checkpointer=checkpointer,
    prompt=agent_prompt_template(),
)


# initialize application server
app = FastAPI()
print("Application started successfully.")
thread_id = uuid4()
config = {"configurable": {"thread_id": thread_id}}


@app.post("/query")
async def query_agent(request: AgentRequest):
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]},
            config=config,
        )
        answer = result["messages"][-1].content.strip()
        return {"result": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/clear_memory")
async def clear_memory():
    try:
        checkpointer.delete_thread(thread_id=thread_id)
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/add_docs")
async def add_documents(request: AddDocsRequest):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in request.documents
        ]
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vectorstore.add_documents(documents=docs, ids=uuids)
        return {"message": "add documents successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
