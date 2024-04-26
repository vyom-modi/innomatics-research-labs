import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="DocSight", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
## Unlock Hidden Insights from Your Documents

This AI-powered chatbot is designed to extract valuable information from your PDF documents. By leveraging Google's Generative AI model Gemini-PRO, it breaks down documents into manageable chunks, creates a searchable vector store, and generates accurate answers to your questions.

### Get Started

Follow these simple steps to unlock insights from your documents:

1. **Enter Your API Key**: Obtain your Google API key from https://aistudio.google.com/app/apikey and enter it below.

2. **Upload Your Documents**: Select multiple PDF files to analyze and gain comprehensive insights.

3. **Ask a Question**: Once your documents are processed, ask a question related to the content and get a precise answer.
""")

with st.sidebar:
    st.title("DocSight Menu:")
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")
    st.markdown("### Upload and Analyze Documents")
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, key="pdf_uploader")
    process_button = st.button("Analyze Documents", key="process_button")

def extract_document_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

def create_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    documents = [Document(chunk) for chunk in text_chunks]
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    return vector_store

def load_conversational_model(api_key):
    prompt_template = """
    Provide a detailed answer based on the context, ensuring all relevant information is included. If the answer is not available in the context, please say so.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_answer(user_question, api_key):
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key))
    docs = vector_store.similarity_search(user_question)
    model = load_conversational_model(api_key)
    response = model({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

if process_button and api_key:  # Check if API key is provided before processing
    with st.spinner("Analyzing Documents..."):
        raw_text = extract_document_text(pdf_docs)
        text_chunks = split_text_into_chunks(raw_text)
        create_vector_store(text_chunks, api_key)
        st.success("Documents Analyzed Successfully!")

st.header("Document Insights Chatbot💡")
user_question = st.text_input("Ask a Question about Your Documents", key="user_question")

if user_question and api_key:  # Ensure API key and user question are provided
    generate_answer(user_question, api_key)