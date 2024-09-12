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

#LLM
api_key = "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"

llm = ChatGroq(groq_api_key = api_key, model_name = 'llama-3.1-70b-versatile', temperature = 0.2, top_p = 0.2)

#Embedding
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import requests
from bs4 import BeautifulSoup

sitemap_urls = ["https://www.bajajallianzlife.com/sitemap.xml", "https://www.hdfclife.com/sitemap.xml", "https://www.tataaia.com/sitemap.xml"]

filter_urls = ["saral", "pension"]

all_urls = []
filtered_urls = []

for sitemap_url in sitemap_urls:
  response = requests.get(sitemap_url)
  sitemap_content = response.content

  #Parse sitemap URL
  soup = BeautifulSoup(sitemap_content, 'xml')
  urls = [loc.text for loc in soup.find_all('loc')]

  #Filter URLs
  selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]

  #Append URLs to the main list
  filtered_urls.extend(selected_urls)

  for url in filtered_urls:
    try:
      print(f"Loading URL: {url}")
      loader = WebBaseLoader(url)
      docs = loader.load()
      loaded_docs.extend(docs)
      print("Successfully loaded content")
    except Exception as e:
      print("Error")

print(f"Loaded urls: {len(filtered_urls)}")

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

#Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 100,
    length_function = len,
)

document_chunks = text_splitter.split_documents(docs)

#Vector database storage
vector_db = FAISS.from_documents(document_chunks, hf_embedding)

query = "Compare the features between the Saral Pension policies offered by the three companies"
docs = vector_db.similarity_search(query)

#Stuff Document Chain Creation
document_chain = create_stuff_documents_chain(llm, prompt)

#Retriever from Vector store
retriever = vector_db.as_retriever()

#Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever,document_chain)

response = retrieval_chain.invoke({"input": "Compare the features between the Saral Pension policies offered by the three companies"})
print(response)
print(response['answer'])
