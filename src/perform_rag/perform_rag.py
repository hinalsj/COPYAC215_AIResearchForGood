import os
import sys
import json
import sqlite3
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
from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden

def get_project_root():
    """
    Find the root directory of the project.
    Assumes the script is running from frontend_ui/ and the project root is two levels up.
    """
    current_file = os.path.abspath(__file__)
    frontend_ui_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(os.path.dirname(frontend_ui_dir))
    return project_root

def resolve_path(relative_path):
    """
    Resolve a path relative to the project root.
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

def debug_print(message):
    """Debug print that works in both Streamlit and standard Python environments"""
    print(message)
    try:
        st.write(message)
    except:
        pass

def retrieve_metadata(source, metadata_file):
    """Retrieve metadata (title, summary, authors) for a paper from the metadata JSON file."""
    try:
        debug_print(f"Attempting to read metadata from: {metadata_file}")
        with open(metadata_file, "r", encoding='utf-8') as f:
            metadata = json.load(f)

        for paper in metadata:
            id = source.strip('.txt').strip('/tmp/')
            if id in paper["paper_id"]:
                return paper["title"], paper["summary"], paper["authors"], paper["paper_id"]
        
        debug_print(f"No metadata found for source: {source}")
        return None, None, None, None
    except FileNotFoundError:
        debug_print(f"Metadata file not found: {metadata_file}")
        debug_print(f"Current working directory: {os.getcwd()}")
        debug_print(f"Resolved metadata file path: {os.path.abspath(metadata_file)}")
        return None, None, None, None
    except Exception as e:
        debug_print(f"Error retrieving metadata: {e}")
        return None, None, None, None

def retrieve_documents(query, persist_directory, model_name, metadata_file):
    try:
        debug_print(f"Initializing embeddings with model: {model_name}")
        hf = HuggingFaceEmbeddings(model_name=model_name)
        
        debug_print(f"Attempting to load Chroma collection from: {persist_directory}")
        db = Chroma(
            collection_name="all_manuscripts",
            embedding_function=hf,
            persist_directory=persist_directory
        )
        
        # Check the number of documents in the collection
        debug_print(f"Total documents in collection: {db._collection.count()}")
        
        # List all documents in the collection for debugging
        if db._collection.count() > 0:
            debug_print("Sample document metadata:")
            for doc in db._collection.get(include=['metadatas'])['metadatas'][:5]:
                debug_print(doc)
        
        debug_print(f"Performing similarity search with query: {query}")
        results = db.similarity_search(query, k=5)
        
        debug_print(f"Number of search results: {len(results)}")
        
        documents = []
        for result in results:
            source = result.metadata.get('source', 'Unknown Source')
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
            else:
                debug_print(f"No metadata found for document with source: {source}")
        
        debug_print(f"Total documents after metadata retrieval: {len(documents)}")
        return documents
    except Exception as e:
        debug_print(f"Error in retrieve_documents: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return []

def download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds):
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.bucket(bucket_name)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    blobs = bucket.list_blobs(prefix=folder_prefix)
    file_count = 0
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, folder_prefix)
        local_path = os.path.join(destination_folder, relative_path)
        local_folder = os.path.dirname(local_path)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        blob.download_to_filename(local_path)
        debug_print(f"Downloaded {blob.name} to {local_path}")
        file_count += 1
    
    debug_print(f"Total files downloaded: {file_count}")
    return file_count

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
    if not documents:
        debug_print("No documents to generate answer from.")
        return "", "No relevant documents found for the given query."

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
    debug_print("Generated response:")
    debug_print(response.text)
    return prompt, response.text

def main(query):
    # Add error handling and more detailed debugging
    try:
        # Resolve paths
        project_root = get_project_root()
        debug_print(f"Project root directory: {project_root}")

        # Resolve file paths
        metadata_file = resolve_path('src/perform_rag/arxiv_social_impact_papers.json')
        destination_folder = resolve_path('paper_vector_db')
        persist_directory = resolve_path('paper_vector_db_local')
        
        # Verify paths
        debug_print(f"Metadata file path: {metadata_file}")
        debug_print(f"Destination folder path: {destination_folder}")
        debug_print(f"Persist directory path: {persist_directory}")

        # Ensure paths exist
        os.makedirs(destination_folder, exist_ok=True)
        os.makedirs(persist_directory, exist_ok=True)

        # Verify metadata file exists
        if not os.path.exists(metadata_file):
            debug_print(f"ERROR: Metadata file {metadata_file} not found!")
            return "Metadata file not found. Cannot proceed."

        # Load credentials
        debug_print("Loading credentials...")
        info = json.loads(st.secrets['secrets_str_1'])
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(info)
        
        # Bucket settings
        bucket_name = 'paper-rec-bucket'
        folder_prefix = 'paper_vector_db/'
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        PROJECT_ID = "ai-research-for-good"
        LOCATION = "us-central1"
        MODEL_ID = "gemini-1.5-flash"
        MODEL_ENDPOINT = "projects/129349313346/locations/us-central1/endpoints/3319822527953371136"

        # Download files from bucket
        file_count = download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds)
        if file_count == 0:
            debug_print("WARNING: No files downloaded from the bucket!")

        # Perform document retrieval
        documents = retrieve_documents(query, persist_directory, model_name, metadata_file)

        if not documents:
            debug_print("No documents found through semantic search.")
            return "No relevant documents found for the given query."

        # Generate answer
        _, answer = generate_answer_google(
            documents, query, PROJECT_ID, LOCATION, MODEL_ID, creds
        )
        return answer

    except Exception as e:
        debug_print(f"Critical error in main function: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    main()
