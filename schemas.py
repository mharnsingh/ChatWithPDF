from pydantic import BaseModel

class AgentRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    page_content: str
    metadata: dict

class AddDocsRequest(BaseModel):
    documents: list[DocumentRequest]
