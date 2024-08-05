from pathlib import Path
from langchain import hub
from langchain.chains import LLMChain
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

# load_dotenv()
# llm =ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"))
# memory=ConversationBufferMemory(memory_key="Chat_history",k=5)
# llm_chain=LLMChain(llm=llm,
#                    )
st.set_page_config(page_title="Chatbot with RAG")
st.title("Configure your Broadband Services:")


if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {"role":"assistant","content":"Hello! How can I help you."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
user_prompt =st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role":"assistant","content":user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
        
