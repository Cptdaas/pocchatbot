import streamlit as st
from pathlib import Path
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
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

load_dotenv()

# Initialize Streamlit app layout
st.set_page_config(page_title="RAG Chatbot")
st.title("Configure your Broadband Services:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Directories
TMP_DIR = "C:/prpwork/01AT&TPOC/data"
LOCAL_VECTOR_STORE_DIR = "C:/prpwork/01AT&TPOC/data/vector_store"
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to load documents
def load_documents():
    loader = CSVLoader("pocdata.csv")
    documents = loader.load()
    return documents

# Function to split documents
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create embeddings and save to local vector store
def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(), persist_directory=LOCAL_VECTOR_STORE_DIR)
    vectordb.persist()
    retriever = vectordb.as_retriever()
    return retriever

# Function to query the LLM
def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    return result['answer']

# Load and process documents if not already done
if 'retriever' not in st.session_state:
    documents = load_documents()
    texts = split_documents(documents)
    retriever = embeddings_on_local_vectordb(texts)
    st.session_state.retriever = retriever

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
prompt = st.chat_input("What is your query?")

if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Query the LLM using the RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            if 'retriever' in st.session_state:
                ai_response = query_llm(st.session_state.retriever, prompt)
                st.markdown(ai_response)
            else:
                st.error("Please load and process documents first.")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Display chat history
st.header("Chat History")
for i, message in enumerate(st.session_state.messages):
    st.write(f"{message['role'].capitalize()} {i+1}: {message['content']}")
