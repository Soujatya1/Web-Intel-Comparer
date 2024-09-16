import streamlit as st
import aiohttp
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import hashlib
import time
import pickle

# Streamlit app layout
st.title("Insurance Policy Comparison Chatbot")

# API Key for LLM
api_key = "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"

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

# Asynchronous function to load documents
async def fetch_document(session, url):
    try:
        async with session.get(url) as response:
            content = await response.text()
            soup = BeautifulSoup(content, 'lxml')
            return soup
    except Exception as e:
        st.error(f"Failed to load content from {url}: {e}")
        return None

async def load_sitemaps(sitemap_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_document(session, url) for url in sitemap_urls]
        return await asyncio.gather(*tasks)

# Step 3: Parallel loading of filtered URLs
def filter_urls(soup, filter_keywords):
    urls = [loc.text for loc in soup.find_all('loc')]
    return [url for url in urls if any(keyword in url for keyword in filter_keywords)]

async def load_filtered_documents(sitemap_urls, filter_keywords):
    sitemaps = await load_sitemaps(sitemap_urls)
    filtered_urls = []
    for sitemap in sitemaps:
        if sitemap:
            filtered_urls.extend(filter_urls(sitemap, filter_keywords))
    
    return filtered_urls

# Step 4: Button to trigger document loading
if st.button("Load Documents"):
    if not sitemap_urls or not filter_keywords:
        st.error("Please enter valid sitemap URLs and keywords.")
    else:
        with st.spinner("Loading documents..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            filtered_urls = loop.run_until_complete(load_filtered_documents(sitemap_urls, filter_keywords))
            if filtered_urls:
                st.success(f"Filtered {len(filtered_urls)} URLs based on keywords.")
            else:
                st.error("No URLs matched the provided keywords.")

# Step 5: Create or load vector store (use caching)
if "filtered_urls" and "vector_db" not in st.session_state:
    try:
        st.session_state.docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        
        # Process URLs in batches
        for url in filtered_urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            st.session_state.docs.extend(docs)
        
        document_chunks = text_splitter.split_documents(st.session_state.docs)
        
        # Cache embeddings and vector store
        with st.spinner("Creating vector store..."):
            embeddings_cache_file = 'embeddings_cache.pkl'
            if not st.session_state.get("vector_db"):
                # Try loading the cached embeddings
                try:
                    with open(embeddings_cache_file, 'rb') as f:
                        st.session_state.vector_db = pickle.load(f)
                    st.success("Loaded vector store from cache.")
                except FileNotFoundError:
                    # If cache not available, create a new one
                    st.session_state.vector_db = FAISS.from_documents(document_chunks, st.session_state.hf_embedding)
                    with open(embeddings_cache_file, 'wb') as f:
                        pickle.dump(st.session_state.vector_db, f)
                    st.success("Vector store created and cached successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Step 6: Querying and displaying results
if "vector_db" in st.session_state:
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
        with st.spinner("Processing your query..."):
            try:
                response = retrieval_chain.invoke({"input": user_query})
                if response and 'answer' in response:
                    st.write(response['answer'])
                else:
                    st.error("No answer returned. Please check the query or data.")
            except Exception as e:
                st.error(f"Error processing query: {e}")
