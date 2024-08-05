from pathlib import Path
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import csv_loader,CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

import streamlit as st

load_dotenv()
llm =ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"))
TMP_DIR ="C:/prpwork/01AT&TPOC/data"
LOCAL_VECTOR_STORE_DIR = "C:/prpwork/01AT&TPOC/data/vector_store"

st.set_page_config(page_title="RAG")
st.title("Configure your Broadband Services:")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def load_documents():
    loader =CSVLoader("pocdata.csv")
    documents=loader.load()
    return documents
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb =Chroma.from_documents(texts,embedding=OpenAIEmbeddings(),persist_directory="C:/prpwork/01AT&TPOC/data")
    vectordb.persist()
    retriever = vectordb.as_retriever()
    return retriever

# Define prompt template for better results
# prompt_template = PromptTemplate(
#     input_variables=["question", "chat_history"],
#     template="""
#     You are a helpful assistant specializing in broadband plans. Answer the following question based on the provided documents.
    
#     Question: {question}
    
#     Chat History: {chat_history}
    
#     Provide a concise and informative response.
#     """
# )
def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result


# st.header("Document Loading and Embedding")
# if st.button("Load and Process Documents"):
documents = load_documents()
texts = split_documents(documents)
retriever = embeddings_on_local_vectordb(texts)
st.session_state.retriever = retriever
st.success("Documents loaded and processed!")

st.header("Query the LLM")
query = st.text_input("Enter your question:")
if st.button("Submit Query"):
    if 'retriever' in st.session_state:
        answer = query_llm(st.session_state.retriever, query)
        st.write(f"Answer: {answer}")
    else:
        st.error("Please load and process documents first.")

st.header("Chat History")
for i, (question, answer) in enumerate(st.session_state.messages):
    st.write(f"Q{i+1}: {question}")
    st.write(f"A{i+1}: {answer}")
