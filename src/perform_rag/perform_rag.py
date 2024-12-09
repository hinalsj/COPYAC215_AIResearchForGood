import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
from google.cloud import storage
import json
from vertexai.generative_models import GenerativeModel
import vertexai

# Configure Chroma connection
configuration = {
    "client_type": "PersistentClient",
    "path": "paper_vector_db/"  # Matches the directory used in embed_papers.py
}

collection_name = "all_manuscripts"

# Initialize Chroma connection
conn = st.experimental_connection("chromadb", type=ChromadbConnection, **configuration)

# Google Cloud setup
def download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds):
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


# Generate answer with Vertex AI
def generate_answer_google(documents, query, project_id, location, model_id, creds):
    structured_docs = "\n\n".join(
        f"""Title: {doc['documents']}
        Metadata: {doc['metadatas']}
        """
        for _, doc in documents.iterrows()
    )
    
    prompt = f"""\nYou are a helpful assistant working for Global Tech Colab For Good, an organization that helps connect non-profit organizations to relevant technical research papers. 
            The following is a query from the non-profit:
            {query}
            We have retrieved the following papers that are relevant to this non-profit's request query. 
            {structured_docs}
            Your job is to provide in a digestible manner the title of the paper(s), their summaries, their metadata, and how the papers can be used by the non-profit to help with their query. 
            Ensure your answer is structured, clear, and user-friendly.
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

    PROJECT_ID = "ai-research-for-good"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-flash"

    # Download vector database if needed
    download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds)

    # Retrieve documents from Chroma
    query_df = conn.retrieve(collection_name=collection_name, query=query)
    st.write(f"Retrieved {len(query_df)} documents")

    # Generate answer using Vertex AI
    prompt, answer = generate_answer_google(query_df, query, PROJECT_ID, LOCATION, MODEL_ID, creds)
    return answer


if __name__ == "__main__":
    query = st.text_input("Enter your query:", "")
    if st.button("Submit"):
        with st.spinner("Fetching relevant papers and generating explanation..."):
            try:
                answer = main(query)
                st.success("Explanation generated successfully!")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
