import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import requests

# Streamlit app title
st.title("Life Insurance Policy Comparison Chatbot")

# LLM configuration
api_key = "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"
llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)

# Embedding configuration
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# User inputs for sitemap URLs and filter keywords
sitemap_input = st.text_area("Enter Sitemap URLs (comma-separated):")
filter_input = st.text_area("Enter Filter Keywords (comma-separated):")

# Process user inputs
sitemap_urls = [url.strip() for url in sitemap_input.split(",") if url.strip()]
filter_urls = [keyword.strip() for keyword in filter_input.split(",") if keyword.strip()]

# Asynchronous fetching of sitemap content
async def fetch_sitemap(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Parse the sitemap content and filter URLs
def parse_sitemap(content, filter_urls):
    soup = BeautifulSoup(content, 'xml')
    urls = [loc.text for loc in soup.find_all('loc')]
    return [url for url in urls if any(filter in url for filter in filter_urls)]

# Fetch and load the documents from the sitemaps asynchronously
async def fetch_documents(sitemap_urls, filter_urls):
    tasks = [fetch_sitemap(url) for url in sitemap_urls]
    sitemap_contents = await asyncio.gather(*tasks)
    
    # Parse sitemap URLs
    all_filtered_urls = []
    for content in sitemap_contents:
        filtered_urls = parse_sitemap(content, filter_urls)
        all_filtered_urls.extend(filtered_urls)
    
    return all_filtered_urls

# Load documents from URLs using threading for parallel processing
def load_documents_parallel(urls):
    loaded_docs = []

    def load_single_url(url):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            return docs
        except Exception as e:
            st.error(f"Error loading document from {url}: {e}")
            return []
    
    # Use a ThreadPoolExecutor for parallel document loading
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(load_single_url, urls))
        for result in results:
            loaded_docs.extend(result)

    return loaded_docs

# Batched embedding creation
def batch_embed_and_store(documents, batch_size=16):
    document_chunks = text_splitter.split_documents(documents)
    
    # Embedding in batches
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i + batch_size]
        vector_db.add_documents(batch)

# Main process for document loading and query handling
if st.button("Load Documents"):
    # Asynchronously fetch the filtered URLs
    filtered_urls = asyncio.run(fetch_documents(sitemap_urls, filter_urls))
    
    st.write(f"Total filtered URLs: {len(filtered_urls)}")

    # Load documents in parallel
    loaded_docs = load_documents_parallel(filtered_urls)
    st.write(f"Loaded documents: {len(loaded_docs)}")

    # Text Splitting and batched vector storing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    vector_db = FAISS(document_chunks, hf_embedding)

    batch_embed_and_store(loaded_docs)

    # ChatPrompt Template creation
    prompt = ChatPromptTemplate.from_template("""
    You are a Life Insurance specialist who needs to answer queries based on the information provided in the websites only. Please follow all the websites, and answer as per the same.
    Do not answer anything except from the website information which has been entered.
    Do not skip any information from the context. Answer appropriately as per the query asked.
    Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the websites, if asked.
    Generate tabular data wherever required to classify the difference between different parameters of policies.
    I will tip you with a $1000 if the answer provided is helpful.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever from Vector store
    retriever = vector_db.as_retriever()

    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input for query
    user_query = st.text_input("Enter your query:", "What are the premium payment modes available for TATA life insurance plan & HDFC Life insurance plan?")

    if st.button("Get Answer"):
        # Perform similarity search and retrieve results based on user input
        response = retrieval_chain.invoke({"input": user_query})

        # Display the result
        st.write("Answer:")
        st.write(response['answer'])

        # Optional: Display relevant sources
        st.write("Sources:")
        for doc in response.get('source_documents', []):
            st.write(f"- {doc.metadata['source']}")
