import os, sqlite3
import json
import tempfile
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import os
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden


def download_files_from_bucket(bucket_name, folder_prefix, destination_folder,creds):
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs(prefix=folder_prefix)
    print("Ran this successfully")
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, folder_prefix)
        local_path = os.path.join(destination_folder, relative_path)
        local_folder = os.path.dirname(local_path)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def retrieve_documents(query, persist_directory, model_name):
    hf = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(
        collection_name="all_manuscripts",
        embedding_function=hf,
        persist_directory=persist_directory
    )

    results = db.similarity_search(query, k=5)
    documents = []
    for result in results:
        source = result.metadata['source']
        page_content = result.page_content
        prompt = f"\nPage Content: {page_content}\n"
        documents.append(prompt)

    return documents

def generate_answer_google(documents, query, project_id, location, model_id, creds):
    documents_combined = "\n\n".join(documents)
    prompt = f"""\nYou are a helpful assistant working for Global Tech Colab For Good, an organization that helps connect non-profit organizations to relevant technical research papers. 
            The following is a query from the non-profit:
            {query}
            We have retrieved the following chunks of research papers that are relevant to this non-profit's request query. 
            {documents_combined}
            Your job is to provide in a digestible manner the title of the paper(s) retrieved and an explanation for how the paper(s) can be used by the non-profit to help with their query. 
            If the title isn't available, make up a relevant title. Even if the papers dont seem useful to the query, do not say that. Try to be as useful to the non-profit and remember that they are the reader of your response."""

    vertexai.init(project=project_id, location="us-central1", credentials=creds)

    model = GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
       prompt
    )

    print(response.text)
    return response.text

def main(query):
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../../secrets/ai-research-for-good-b6f4173936f9.json"
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =  st.secrets

    #secrets_dict = dict(st.secrets)

    # Fix formatting for private_key if necessary
    #if "private_key" in secrets_dict:
        #secrets_dict["private_key"] = secrets_dict["private_key"].replace("\\n", "\n")
    
    # Convert the dictionary to JSON
    #info = json.dumps(secrets_dict, indent=4)
    
    #info = json.loads(st.secrets)
    #creds = service_account.Credentials.from_service_account_info(secrets_dict)
    st.write("hello")
    info = json.loads(st.secrets['secrets_str_1'])
    st.write(info)
    info["private_key"] = info["private_key"].replace("\\n", "\n")

    creds = service_account.Credentials.from_service_account_info(info)
    
    bucket_name = 'paper-rec-bucket'
    destination_folder = 'paper_vector_db'
    folder_prefix = 'paper_vector_db/'
    persist_directory = 'paper_vector_db/'
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    PROJECT_ID = "ai-research-for-good"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-pro"

    #query = "AI for social impact"

    download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds)
    documents = retrieve_documents(query, persist_directory, model_name)
    # print(documents)
    answer = generate_answer_google(documents, query, PROJECT_ID, LOCATION, MODEL_ID, creds)

    return answer

if __name__ == "__main__":
    main()
