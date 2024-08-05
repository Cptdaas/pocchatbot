from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import csv_loader,CSVLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI, OpenAIChat
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app layout
st.set_page_config(page_title="Chatbot with RAG")
st.title("Configure your Broadband Services:")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Directories
TMP_DIR = Path("C:/prpwork/01AT&TPOC/data")
LOCAL_VECTOR_STORE_DIR = Path("C:/prpwork/01AT&TPOC/data/vector_store")
llm =ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to load documents
def load_documents():
    loader =CSVLoader("pocdata.csv")
    documents=loader.load()
    return documents

# Function to split documents
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to create embeddings and save to local vector store
def embeddings_on_local_vectordb(texts):
    vectordb =Chroma.from_documents(texts,embedding=OpenAIEmbeddings(),persist_directory="C:/prpwork/01AT&TPOC/data")
    vectordb.persist()
    retriever = vectordb.as_retriever()
    return retriever
# Define prompt template for better results
prompt_template = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
    You are a helpful assistant specializing in broadband plans. Answer the following question based on the provided documents.
    
    Question: {question}
    
    Chat History: {chat_history}
    
    Provide a concise and informative response.
    """
)

# Function to query the LLM
def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True,
        prompt=prompt_template
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

# Display document loading section
st.header("Document Loading and Embedding")

if st.button("Load and Process Documents"):
    documents = load_documents()
    texts = split_documents(documents)
    retriever = embeddings_on_local_vectordb(texts)
    st.session_state.retriever = retriever
    st.success("Documents loaded and processed!")
else:
    st.warning("Please enter the OpenAI API key.")
# documents = load_documents()
# texts = split_documents(documents)
# retriever = embeddings_on_local_vectordb(texts)
# st.session_state.retriever = retriever

# Display query submission section
# st.header("Chat with the Bot")
query = st.text_input("Enter your question:")
if st.button("Submit Query"):
    if 'retriever' in st.session_state:
        answer = query_llm(st.session_state.retriever, query)
        st.write(f"Answer: {answer}")
    else:
        st.error("Please load and process documents first.")
        
# Display chat history
st.header("Chat History")
for i, (question, answer) in enumerate(st.session_state.messages):
    st.write(f"Q{i+1}: {question}")
    st.write(f"A{i+1}: {answer}")
