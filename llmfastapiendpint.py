from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Directories
TMP_DIR = "C:/prpwork/01AT&TPOC/data"
LOCAL_VECTOR_STORE_DIR = "C:/prpwork/01AT&TPOC/data/vector_store"
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to load documents
def load_documents():
    loader = CSVLoader("pocdata.csv")
    documents = loader.load()
    return documents

# Function to split documents
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create embeddings and save to local vector store
def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(), persist_directory=LOCAL_VECTOR_STORE_DIR)
    retriever = vectordb.as_retriever()
    return retriever

# Define the prompt template
prompt_template = """
You are an Internet broadband plan provider. Please provide the all possible plan for internet to the user based on the following provided context:
the question requested configurtion in not fit compatible. show the user it is not compatible. suggest the releted suitable plan
Think step by step and list all configuration plans in a table format.
also create discription of above plan for user.If user is asked for other plan, please reccomend him other plan.
If user asking for Quote gereneration for specific configuration plan, then only create and configuration plan 
description in json format.<context>
{context}
</context>
Question: {input}
If you don't know,Show ask Survice Provider for more details.
"""
prompt1 = ChatPromptTemplate.from_template(prompt_template)

# Function to query the LLM
def query_llm(retriever, query):
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt1
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(query)
    return result

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    global retriever
    documents = load_documents()
    texts = split_documents(documents)
    retriever = embeddings_on_local_vectordb(texts)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if 'retriever' not in globals():
        raise HTTPException(status_code=500, detail="Retriever not initialized.")
    ai_response = query_llm(retriever, request.query)
    return {"response": ai_response}


