from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from llama_index.llms.gemini import Gemini
import os
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
import faiss
import streamlit as st
from api_keys import GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Utilities
def pickle_loader(file_name, all_files):
    file_name = f"{file_name}.pkl"
    if file_name in all_files:
        with open(file_name, "rb") as f:
            return pickle.load(f)
    return None

def pickle_saver(obj, file_name):
    file_name = f"{file_name}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)

def get_vector_store(all_files):
    vector_store = pickle_loader("vector-store", all_files)
    if not vector_store: vector_store = faiss.IndexFlatL2(768)
    return vector_store

def get_database(all_files):
    if "database.csv" in all_files: return pd.read_csv("database.csv", index_col = "Unnamed: 0")
    return pd.DataFrame(columns = ["url", "text", "timestamp"])

def get_encoder(all_files):
    encoder = pickle_loader("encoder", all_files)
    if not encoder:
        encoder = SentenceTransformer("all-mpnet-base-v2")
        pickle_saver(encoder, "encoder")
    return encoder

def save_vector_store(vector_store):
    pickle_saver(vector_store, "vector-store")

def save_df(df):
    df.to_csv("database.csv")

def ingest_chunks(chunks, vector_store, df, encoder):
    chunk_texts = []
    chunk_rows = []
    for chunk in chunks:
        chunk_text = chunk.page_content.strip()
        if chunk_text:
            dt = datetime.now()
            chunk_url = chunk.metadata["source"]
            chunk_rows.append([chunk_url, chunk_text, dt])
            chunk_texts.append(chunk_text)
    chunk_df = pd.DataFrame(chunk_rows, columns = ["url", "text", "timestamp"])
    df = pd.concat([df, chunk_df], ignore_index = True)
    chunk_embeddings = encoder.encode(chunk_texts)
    vector_store.add(chunk_embeddings)
    save_vector_store(vector_store)
    save_df(df)

    return vector_store, df

def get_answer(query, vector_store, df, encoder, gemini_model):
    encoded_query = encoder.encode([query])
    try:
        relevant_chunk_indices = vector_store.search(encoded_query, 3)[1][0]
        print(relevant_chunk_indices)
    except:
        print("No relevant chunk found")

    chunk_responses = []
    chunk_urls = []
    for i in relevant_chunk_indices:
        relevant_chunk = df.loc[i, "text"]
        relevant_url = df.loc[i, "url"]
        prompt = f"Answer the question in double quotes using the text in backticks (`) only if the text is relevant to the question else return 'No answer':\n\"{query}\"\n`{relevant_chunk}`"
        chunk_response = str(gemini_model.complete(prompt))
        if not chunk_response == "No answer":
            chunk_responses.append(chunk_response)
            chunk_urls.append(relevant_url)

    if chunk_responses:
        summarization_prompt = f"Summarize in detail: {" ".join(chunk_responses)}"
        answer = gemini_model.complete(summarization_prompt)

    return answer, chunk_urls


# Application UI using Streamlit
def main(vector_store, df, encoder, document_splitter, gemini_model):
    st.title("URL QA Tool Using Google Gemini")
    st.sidebar.title("URL Ingester")

    urls = []
    for i in range(1, 4):
        url = st.sidebar.text_input(f"Enter URL #{i}")
        url = url.strip()
        if url: urls.append(url)
    is_ingest_clicked = st.sidebar.button("Ingest URLs")

    if is_ingest_clicked:
        with st.sidebar:
            with st.spinner("Ingesting URLs"):
                if urls:
                    url_loader = UnstructuredURLLoader(urls = urls)
                    documents = url_loader.load()
                    chunks = document_splitter.split_documents(documents)
                    vector_store, df = ingest_chunks(chunks, vector_store, df, encoder)
                    st.sidebar.success("Ingestion successful!")
                    print("Ingestion successful!")
                else:
                    st.sidebar.warning("Please add at least one URL")
                    print("No URL entered")
    
    query = st.text_area("Write your query with respect to the recently or previously ingested URLs")
    is_query_clicked = st.button("Ask")

    if is_query_clicked:
        if query:
            with st.spinner("Asking Google Gemini"):
                try:
                    answer, urls = get_answer(query, vector_store, df, encoder, gemini_model)
                except:
                    answer = "No answer"
                    urls = []
            st.markdown(f"### Answer\n{answer}")
            if urls:
                st.markdown(f"### Sources")
                for url in set(urls):
                    st.markdown(f"- {url}")
            st.success("Answered!")
            print(answer)
        else:
            st.warning("Please write your question")
            print("Please write your question")

if __name__ == "__main__":
    all_files = os.listdir()
    vector_store = get_vector_store(all_files)
    df = get_database(all_files)
    encoder = get_encoder(all_files)
    document_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", " "], chunk_size = 1000, chunk_overlap = 200)
    gemini_model = Gemini(model = "models/gemini-pro")

    main(vector_store, df, encoder, document_splitter, gemini_model)
    