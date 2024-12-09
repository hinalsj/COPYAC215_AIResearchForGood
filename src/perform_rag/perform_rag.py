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
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden


def download_files_from_bucket(bucket_name, folder_prefix, destination_folder,creds):
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs(prefix=folder_prefix)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, folder_prefix)
        local_path = os.path.join(destination_folder, relative_path)
        local_folder = os.path.dirname(local_path)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def retrieve_metadata(source, metadata_file="arxiv_social_impact_papers.json"):
    """Retrieve metadata (title, summary, authors) for a paper from the metadata JSON file."""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    for paper in metadata:
        id = source.strip('.txt').strip('/tmp/')
        if id in paper["paper_id"]:
            return paper["title"], paper["summary"], paper["authors"], paper["paper_id"]
    return None, None, None,None


def retrieve_documents(query, persist_directory, model_name, metadata_file="arxiv_social_impact_papers.json"):
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
        
        # Retrieve metadata
        title, summary, authors, url = retrieve_metadata(source, metadata_file)
        if title and summary:
            prompt = {
                "title": title,
                "summary": summary,
                "authors": authors,
                "page_content": page_content,
                "url": url
            }
            documents.append(prompt)

    return documents


def rank_and_filter_documents(query, documents, project_id, location, model_endpoint, creds):
    """
    Rank and filter documents using the fine-tuned model.
    """
    # Use the fine-tuned model's ranking function
    filtered_docs = []
    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel(model_endpoint)

    for doc in documents:
        Input = f"""You are an expert data annotator who works on a project to connect non-profit users to technological research papers that might be relevant to the non-profit's use case.
        Please rate the following research paper for its relevance to the non-profit's user query. Output "Relevant" if the paper relevant, or "Not Relevant" if the paper is not relevant.

        User query: {query}

        Paper title: {doc['title']}
        Paper summary: {doc['summary']}
        """

        response = model.generate_content(Input)
        generated_text = response.text
        if "relevant" in generated_text.lower():
            filtered_docs.append(doc)

    return filtered_docs


def generate_answer_google(documents, query, project_id, location, model_id, creds):
    """
    Generate an answer with structured output combining title, summary, authors, page content, and URL.
    """
    structured_docs = "\n\n".join(
        f"""Title: {doc['title']}
        Summary: {doc['summary']}
        Authors: {', '.join(doc['authors']) if doc['authors'] else 'Unknown'}
        Relevant Chunk: {doc['page_content']}
        Paper URL: {doc['url']}
                """
        for doc in documents
    )
    
    prompt = f"""\nYou are a helpful assistant working for Global Tech Colab For Good, an organization that helps connect non-profit organizations to relevant technical research papers. 
            The following is a query from the non-profit:
            {query}
            We have retrieved the following papers that are relevant to this non-profit's request query. 
            {structured_docs}
            Your job is to provide in a digestible manner the title of the paper(s), their summaries, their URLs, and how the papers can be used by the non-profit to help with their query. 
            Ensure your answer is structured, clear, and user-friendly, and include the Paper URL in your response for each paper.
            """

    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel(model_id)

    response = model.generate_content(prompt)
    print(response.text)
    return prompt, response.text


def main(query):
    info = json.loads(st.secrets['secrets_str_1'])
    info["private_key"] = info["private_key"].replace("\\n", "\n")
    creds = service_account.Credentials.from_service_account_info(info)
    bucket_name = 'paper-rec-bucket'
    destination_folder = 'paper_vector_db'
    folder_prefix = 'paper_vector_db/'
    persist_directory = 'paper_vector_db_local/'
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    PROJECT_ID = "ai-research-for-good"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-flash"
    MODEL_ENDPOINT = "projects/129349313346/locations/us-central1/endpoints/3319822527953371136"

    download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds)

    documents = retrieve_documents(query, persist_directory, model_name)

    # top_documents = rank_and_filter_documents(query, documents, PROJECT_ID, LOCATION, MODEL_ENDPOINT, creds)
    prompt, answer = generate_answer_google(
        documents, query, PROJECT_ID, LOCATION, MODEL_ID, creds
    )
    return answer


if __name__ == "__main__":
    main()
