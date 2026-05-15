import os
import lancedb
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

def search_knowledge_base(query: str) -> str:
    """Searches the LanceDB knowledge base for relevant information about ADK and LanceDB.
    
    Args:
        query: The search query or question.
        
    Returns:
        Relevant excerpts from the knowledge base.
    """
    try:
        # Initialize Vertex AI inside the function to ensure environment variables are loaded
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "demolab-puplampu01")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        bucket_name = "lance-data-adk-01"

        vertexai.init(project=project_id, location=location)
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

        db_uri = f"gs://{bucket_name}/lance-data/knowledge_base"
        db = lancedb.connect(db_uri)
        
        table_name = "adk_docs"
        table = db.open_table(table_name)
        
        # Generate embedding for the search query
        inputs = [TextEmbeddingInput(query, "RETRIEVAL_QUERY")]
        embeddings = embedding_model.get_embeddings(inputs)
        query_vector = embeddings[0].values
        
        # Search the vector database
        results = table.search(query_vector).limit(3).to_list()
        
        if not results:
            return "No specific entries found in the knowledge base for this query."
            
        formatted_results = "\n\n".join([f"- {res['text']}" for res in results])
        return f"### Results from LanceDB Knowledge Base:\n{formatted_results}"
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
