from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from pydantic import BaseModel, Field
from collections import defaultdict
from typing import List


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )


def format_citation(docs: List[Document], citation: List[int]) -> str:
    source_pages = defaultdict(set)

    for idx in citation:
        if idx < 0 or idx >= len(docs):
            continue
        doc = docs[idx]
        source, page = doc.metadata["source"], doc.metadata["page"]
        source_pages[source].add(page)

    if not source_pages:
        return "No references available."

    formatted = []
    for i, (source, pages) in enumerate(source_pages.items(), start=1):
        sorted_pages = sorted(pages)
        pages_str = ", ".join(str(p) for p in sorted_pages)
        formatted.append(f"Source {i}: {source}; Page: {pages_str}")

    return "\n".join(formatted)


def format_docs(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n".join(formatted)


def qa_prompt_template():

    system_prompt = """\
    You are a helpful assistant designed to answer user questions based strictly on provided source documents.

    Answer the user's question using only the retrieved context below. 
    If the answer cannot be found or clearly inferred from the context, respond with:
    "I’m sorry, the information you’re asking for is not available in the provided documents."

    Guidelines:
    - Do not assume or invent information that is not present in the context.
    - Cite specific details from the context when relevant.

    Context:
    {context}\
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    return prompt


def agent_prompt_template():
    agent_prompt = '''\
You are an AI agent with a tool that retrieves relevant documents and generates an answer with citations.

1. Review the chat history and context to craft the most effective query.  
2. If the user’s query contains multiple parts, tasks, or questions, break it down into clear, concise sub-queries that can each be answered independently.
3. For each sub-query, call the tool to retrieve documents and generate an answer with citations.  
4. If multiple answers are returned, combine them into a single, coherent, well-structured response. 
5. At the end of the response, merge all citations into a single list using the following format exactly:  
**Source N: [article title]; Page: a, b, c**\
'''
    return agent_prompt

