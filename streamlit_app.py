import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import hashlib

# Streamlit app layout
st.title("Insurance Policy Comparison Chatbot")

# API Key for LLM
api_key = st.secrets["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"]

# Initialize the LLM and Embedding only once
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)

if "hf_embedding" not in st.session_state:
    st.session_state.hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Step 1: User input
sitemap_urls_input = st.text_area("Enter Sitemap URLs (one per line)", height=150)
filter_keywords_input = st.text_input("Enter Keywords (comma-separated)")
sitemap_urls = [url.strip() for url in sitemap_urls_input.split("\n") if url.strip()]
filter_keywords = [keyword.strip() for keyword in filter_keywords_input.split(",") if keyword.strip()]

# Step 2: Hash URLs to cache documents
def get_cache_key(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def load_document_from_url(url):
    # Cached document check
    cache_key = get_cache_key(url)
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        loader = WebBaseLoader(url)
        loaded_docs = loader.load()
        st.session_state[cache_key] = loaded_docs # Cache loaded document
        return loaded_docs
    except Exception as e:
        st.error(f"Failed to load content from {url}: {e}")
        return []

# Step 3: Parallel loading of filtered URLs
def load_documents_parallel(sitemap_urls, filter_keywords):
    filtered_urls = []
    with ThreadPoolExecutor() as executor:
        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url)
                sitemap_content = response.content
                soup = BeautifulSoup(sitemap_content, 'lxml')
                urls = [loc.text for loc in soup.find_all('loc')]
                selected_urls = [url for url in urls if any(keyword in url for keyword in filter_keywords)]
                filtered_urls.extend(selected_urls)
            except Exception as e:
                st.error(f"Error loading sitemap {sitemap_url}: {e}")

        # Load documents in parallel
        futures = [executor.submit(load_document_from_url, url) for url in filtered_urls]
        docs = []
        for future in futures:
            docs.extend(future.result()) # Combine results from all threads
        return docs

# Step 4: Button to trigger document loading
if st.button("Load Documents"):
    if not sitemap_urls or not filter_keywords:
        st.error("Please enter valid sitemap URLs and keywords.")
    else:
        with st.spinner("Loading documents..."):
            docs = load_documents_parallel(sitemap_urls, filter_keywords)
            if docs:
                st.session_state.docs = docs
                st.session_state.docs_loaded = True
                st.success(f"Loaded {len(docs)} documents successfully.")
            else:
                st.error("No documents loaded. Please check the sitemap URLs and keywords.")

# Step 5: Proceed if documents are loaded
if "docs_loaded" in st.session_state and st.session_state.docs_loaded:
    # Text splitting
    if "document_chunks" not in st.session_state:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        st.session_state.document_chunks = text_splitter.split_documents(st.session_state.docs)

    # Vector database creation
    if "vector_db" not in st.session_state:
        try:
            with st.spinner("Creating vector store..."):
                st.session_state.vector_db = FAISS.from_documents(st.session_state.document_chunks, st.session_state.hf_embedding)
                st.success("Vector store created successfully.")
        except Exception as e:
            st.error(f"Error creating vector store: {e}")

    # User query input and LLM interaction
    prompt = ChatPromptTemplate.from_template(
        """
        You are an HDFC Life Insurance specialist who needs to answer queries based on the information provided in the websites.
        Compare your policies against other companies. Provide tabular data wherever required.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    retriever = st.session_state.vector_db.as_retriever()
    document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_query = st.text_input("Ask a question about the Saral Pension policies")

    if user_query:
        try:
            response = retrieval_chain.invoke({"input": user_query})
            if response and 'answer' in response:
                st.write(response['answer'])
            else:
                st.error("No answer returned. Please check the query or data.")
        except Exception as e:
            st.error(f"Error processing query: {e}")
