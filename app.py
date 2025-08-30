import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline
)

# --- Load environment variables ---
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A with LangChain", layout="wide")
st.title("📄 PDF Q&A using HuggingFace + LangChain")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Save to temp folder
    os.makedirs("uploaded_pdfs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("uploaded_pdfs", file.name), "wb") as f:
            f.write(file.getbuffer())

    # Load & split PDFs
    loader = PyPDFDirectoryLoader("uploaded_pdfs")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)

    # Create HuggingFace embeddings
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Select model
    selected_model = st.selectbox(
        "Choose a model",
        options=["mistralai/Mistral-7B-v0.1", "google/flan-t5-base"],
        index=1  # Default to smaller one for local dev
    )

    # Load tokenizer and model using Transformers directly
    if "mistral" in selected_model:
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model,
            trust_remote_code=True,
            token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            trust_remote_code=True,
            token=hf_token
        )
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.1
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            selected_model,
            token=hf_token
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            selected_model,
            token=hf_token
        )
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.1
        )

    # Wrap in LangChain HuggingFacePipeline
    hf = HuggingFacePipeline(pipeline=hf_pipeline)

    # Setup RetrievalQA chain
    prompt_template = """
    Use the following context to answer the question.
    If you don't know, say "I don't know" instead of making up an answer.

    {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # User query input
    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Searching..."):
            result = retrievalQA.invoke({"query": query})

            st.subheader("Answer:")
            st.write(result["result"])

            # Show source documents
            with st.expander("📌 Source Documents"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:500] + "...")


