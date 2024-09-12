import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

#Setup Streamlit page
st.title("Insurance Policy Intelligence and Comparison Chatbot")

#LLM
api_key = "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"

llm = ChatGroq(groq_api_key = api_key, model_name = 'llama-3.1-70b-versatile', temperature = 0.2, top_p = 0.2)

#Embedding
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Placeholder for user input: Web URLs
st.write("Enter the sitemap URLs:")
sitemap_urls_input = st.text_area("Sitemap URLs", height = 100, placeholder = "Enter URLs like https://example.com/sitemap.xml")

#Placeholder for user input: keywords to filter
st.write("Enter keywords to filter URLs:")
filter_urls_input = st.text_input("Keywords", placeholder = "Enter keywords like 'saral,pension'")

#Parse user input
sitemap_urls = [url.strip() for url in sitemap_urls_input.split("\n") if url.strip()]
filter_urls = [keyword.strip() for keyword in filter_urls_input.split(",") if keyword.strip()]

#URL scraping logic
def load_sitemap(sitemap_urls, filter_urls):
  filtered_urls = []
  for sitemap_url in sitemap_urls:
    try:
      response = requests.get(sitemap_url)
      sitemap_content = response.content

      #Parse sitemap URL
      soup = BeautifulSoup(sitemap_content, 'xml')
      urls = [loc.text for loc in soup.find_all('loc')]

      #Filter URLs
      selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]
    except Exception as e:
      st.error(f"Error loading sitemap: {e}")
return filtered_urls

#Load the URLs when user clicks a button
if st.button("Load Documents"):
  if not sitemap_urls or not filter_urls:
    st.error("Please enter both sitemap URLs and keywords")
  else:
    st.write("Loading and filtering URLs")
    filtered_urls = load_sitemap(sitemap_urls, filter_urls)
    st.write(f"Loaded {len(filtered_urls)} filtered URLs")
    
    docs = []
    for url in filtered_urls:
      try:
        st.write(f"loading content from: {url}")
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
        st.success(f"Successfully loaded content from: {url}")
      except Exception as e:
        st.error("Failed to load content")
    if docs:
      st.write("Splitting documents into chunks...")
      #Text Splitting
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1500,
          chunk_overlap  = 100,
          length_function = len,
      )
      document_chunks = text_splitter.split_documents(docs)

      #Vector db creation
      st.write("Creating vector store...")
      vector_db = FAISS.from_documents(document_chunks, hf_embedding)

      #Craft ChatPrompt Template
      prompt = ChatPromptTemplate.from_template(
      """
      You are an HDFC Life Insurance specialist who needs to answer queries based on the information provided in the websites. Please follow all the websites, and answer as per the same.
      Do not answer anything out of the website information.
      Do not skip any information from the context. Answer appropriately as per the query asked.
      Now, being an excellent HDFC Life Insurance agent, you need to compare your policies against the other company's policies in the websites.
      Generate tabular data wherever required to classify the difference between different parameters of policies.
      I will tip you with a $1000 if the answer provided is helpful.
      <context>
      {context}
      </context>
      Question: {input}""")

      #Retriever from Vector store
      retriever = vector_db.as_retriever()

      #Stuff Document Chain Creation
      document_chain = create_stuff_documents_chain(llm, prompt)

      #Create a retrieval chain
      retrieval_chain = create_retrieval_chain(retriever,document_chain)

      #Process the query
      if user_query:
        st.write(f"Processing query: {user_query}")
        response = retrieval_chain.invoke({"input": user_query})
        st.write(response['answer'])
