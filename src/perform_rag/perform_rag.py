import json

def retrieve_metadata(source, metadata_file="arxiv_social_impact_papers.json"):
    """Retrieve metadata (title, summary, authors) for a paper from the metadata JSON file."""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    for paper in metadata:
        if source in paper["paper_id"]:
            return paper["title"], paper["summary"], paper["authors"]
    return None, None, None


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
        title, summary, authors = retrieve_metadata(source, metadata_file)
        if title and summary:
            prompt = {
                "title": title,
                "summary": summary,
                "authors": authors,
                "page_content": page_content
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
    Generate an answer with structured output combining title, summary, authors, and page content.
    """
    structured_docs = "\n\n".join(
        f"""Title: {doc['title']}
Summary: {doc['summary']}
Authors: {', '.join(doc['authors']) if doc['authors'] else 'Unknown'}
Page Content: {doc['page_content']}
        """
        for doc in documents
    )
    prompt = f"""\nYou are a helpful assistant working for Global Tech Colab For Good, an organization that helps connect non-profit organizations to relevant technical research papers. 
            The following is a query from the non-profit:
            {query}
            We have retrieved the following papers that are relevant to this non-profit's request query. 
            {structured_docs}
            Your job is to provide in a digestible manner the title of the paper(s), their summaries, and how the papers can be used by the non-profit to help with their query. 
            Ensure your answer is structured, clear, and user-friendly.
            """

    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel(model_id)

    response = model.generate_content(prompt)
    print(response.text)
    return response.text


def main(query):
    info = json.loads(st.secrets['secrets_str_1'])
    info["private_key"] = info["private_key"].replace("\\n", "\n")

    creds = service_account.Credentials.from_service_account_info(info)
    bucket_name = 'paper-rec-bucket'
    destination_folder = 'paper_vector_db'
    folder_prefix = 'paper_vector_db/'
    persist_directory = 'gs://paper-rec-bucket/paper_vector_db/'
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    PROJECT_ID = "ai-research-for-good"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-pro"
    MODEL_ENDPOINT = "projects/129349313346/locations/us-central1/endpoints/3319822527953371136"

    download_files_from_bucket(bucket_name, folder_prefix, destination_folder, creds)

    documents = retrieve_documents(query, persist_directory, model_name)

    top_documents = rank_and_filter_documents(query, documents, PROJECT_ID, LOCATION, MODEL_ENDPOINT, creds)

    answer = generate_answer_google(
        top_documents, query, PROJECT_ID, LOCATION, MODEL_ID, creds
    )

    return answer


if __name__ == "__main__":
    main()
