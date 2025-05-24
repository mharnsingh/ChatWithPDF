# Chat With PDF (RAG Application)

This repository contains a Retrieval-Augmented Generation (RAG) application that enables users to chat with a set of PDF research papers. The project demonstrates the integration of LangChain agents, tools, memory, vector search, and LLMs to answer and follow up on user queries directly from ingested PDF content.

## Project Structure

- **agent/**  
  Core backend logic for PDF ingestion, parsing, vector storage, and the RAG agent API.

- **demo/**  
  Streamlit web interface for interacting with the RAG agent.

- **eval/**  
  Contains mock Q&A pairs to support future evaluation pipelines.

- **qdrant_data/**
  Persistent storage for Qdrant vector database.

- nginx.conf
  NGINX configuration for reverse proxying and routing.

- docker-compose.yaml
  Orchestrates all services (agent, demo, qdrant, nginx) for local development.

---

## How Does It Work?

1. **PDF Ingestion**  
   The ingest_pdf.py script parses and chunks PDF files from the specified directory, then stores their embeddings in a Qdrant vector database.

2. **Question Answering**  
   User queries are received via the API or web demo. The agent retrieves relevant document chunks using hybrid (dense + sparse) search and always includes clear citations to the source PDF and page in its answers.

3. **Memory Support**  
   The agent maintains conversational memory to support follow-up questions, enabling both context-aware retrieval and contextually relevant responses.

4. **LangGraph ReAct Agent**  
   The core logic leverages a LangGraph ReAct agent, enabling multi-step reasoning and tool use for handling complex queries.

---

## How to Run Locally (Using Docker Compose)

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/chat-with-pdf-rag.git
cd chat-with-pdf-rag
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root with the following keys:

```
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### 3. Run the Application with Docker Compose

Build and run all services by executing:

```sh
docker-compose up --build
```

This will start the following services:

- **qdrant:** Vector store service for document retrieval.
- **agent:** Chat with PDF backend service (FastAPI).
- **demo:** Streamlit web interface for chatting with the agent.
- **nginx:** Reverse proxy exposing the application on port 80.

## Usage

- **Web Demo:**  
  Open your browser and navigate to `http://localhost` to access the Streamlit demo interface.

- **API Endpoint:**

  - Query:
    Send a POST request to http://localhost/agent/query with a JSON payload:

  ```json
  {
    "query": "How can LLM translate text into SQL?"
  }
  ```

  - Clear Memory:
    Send a POST request to http://localhost/agent/clear_memory to clear the current session memory.

- **Monitoring:**  
   Interactions and traces are monitored via LangSmith. You can view traces at [smith.langchain.com](https://smith.langchain.com).

## Future Improvements

- Evaluation pipeline.
- Multi-user session and memory support.
- Advanced chunking and retrieval strategies.
- PDF upload in demo page for user-specific queries.
