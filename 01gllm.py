import streamlit as st
from langchain_community.document_loaders import csv_loader,CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_csv_text():
    loader =CSVLoader("pocdata.csv")
    doc=loader.load()
    return doc

def get_text_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    texts=text_splitter.split_documents(doc)
    return texts


def get_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectordb =Chroma.from_documents(texts,embedding=embeddings,persist_directory="C:/prpwork/01AT&TPOC/data")
    vectordb.persist()
    retrivers =vectordb.as_retriever(search={'k':100})

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", 
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
