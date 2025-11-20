import os
import shutil
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
import faiss

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)


def store_user_embedding(file_path, user_id, data_source=None, save_path="user_embeddings"):
    """
    Store text embeddings from a file in local paths organized by user_id.

    Args:
        file_path (str): Path to the text file containing content to embed and store
        user_id (str): Unique identifier for the user
        data_source (str, optional): Source/type of the data. If None, extracted from filename
        save_path (str): Base path for storing embeddings (default: "user_embeddings")

    Returns:
        dict: Results including document ID and storage path
    """
    # Extract data source from filename if not provided
    if data_source is None:
        # Extract file extension as data source, or use filename
        file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
        data_source = file_extension if file_extension else os.path.basename(file_path)

    # Check if file exists and read content
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error": f"File not found: {file_path}",
            "message": f"Failed to find file {file_path}"
        }

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error reading file {file_path}: {str(e)}",
            "message": f"Failed to read file {file_path}"
        }

    # Create user-specific directory
    user_dir = os.path.join(save_path, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    # Path for the user's vector store
    vector_store_path = os.path.join(user_dir, f"{user_id}_faiss_index")

    # Create document with metadata
    document = Document(
        page_content=text_content,
        metadata={
            "source": data_source,
            "user_id": user_id,
            "filename": file_path
        }
    )

    try:
        # Try to load existing vector store
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create new vector store if none exists
            index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

        # Generate unique ID for the document
        doc_id = str(uuid4())

        # Add document to vector store
        vector_store.add_documents(documents=[document], ids=[doc_id])

        # Save the updated vector store
        vector_store.save_local(vector_store_path)

        return {
            "status": "success",
            "document_id": doc_id,
            "user_id": user_id,
            "data_source": data_source,
            "storage_path": vector_store_path,
            "message": f"Successfully stored embedding for user {user_id}"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to store embedding for user {user_id}"
        }


def search_user_embeddings(query_text, user_id, k=3, filter_source=None, save_path="user_embeddings"):
    """
    Search for similar embeddings in a user's stored data.

    Args:
        query_text (str): The text to search for
        user_id (str): User ID to search in
        k (int): Number of results to return (default: 3)
        filter_source (str): Filter by data source (optional)
        save_path (str): Base path for storing embeddings (default: "user_embeddings")

    Returns:
        list: Search results with similarity scores
    """
    user_dir = os.path.join(save_path, str(user_id))
    vector_store_path = os.path.join(user_dir, f"{user_id}_faiss_index")

    if not os.path.exists(vector_store_path):
        return {
            "status": "error",
            "message": f"No embeddings found for user {user_id}"
        }

    try:
        # Load the user's vector store
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Set up filter if specified
        filter_dict = {"user_id": user_id}
        if filter_source:
            filter_dict["source"] = filter_source

        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(
            query_text,
            k=k,
            filter=filter_dict
        )

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            })

        return {
            "status": "success",
            "user_id": user_id,
            "query": query_text,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to search embeddings for user {user_id}"
        }


def get_user_info(user_id, save_path="user_embeddings"):
    """
    Get information about a user's stored embeddings.

    Args:
        user_id (str): User ID to get info for
        save_path (str): Base path for storing embeddings (default: "user_embeddings")

    Returns:
        dict: User information and statistics
    """
    user_dir = os.path.join(save_path, str(user_id))
    vector_store_path = os.path.join(user_dir, f"{user_id}_faiss_index")

    if not os.path.exists(vector_store_path):
        return {
            "status": "error",
            "message": f"No embeddings found for user {user_id}"
        }

    try:
        # Load the user's vector store
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Get all documents from the store
        all_docs = []
        for doc_id in vector_store.index_to_docstore_id.values():
            doc = vector_store.docstore.search(doc_id)
            if doc:
                all_docs.append(doc)

        # Analyze data sources
        sources = {}
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        return {
            "status": "success",
            "user_id": user_id,
            "total_documents": len(all_docs),
            "storage_path": vector_store_path,
            "data_sources": sources,
            "directory_size": sum(os.path.getsize(os.path.join(user_dir, f))
                                for f in os.listdir(user_dir)
                                if os.path.isfile(os.path.join(user_dir, f)))
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to get info for user {user_id}"
        }


# Example usage
if __name__ == "__main__":

    # Example: Store embeddings from text files for different users
    # Simple usage - just file path and user_id
    print("Storing embeddings for user1 (auto-detected data source)...")
    result1 = store_user_embedding(
        "2022-05-MAY-PASSPORT-POLVERARIGI.txt",
        "POLVERARIGI"
    )
    print(result1)

    print("\nStoring embeddings for user2 (auto-detected data source)...")
    result2 = store_user_embedding(
        "2022-12-CHECK_IRS-TAX_REFUND_122022-POLVERARIGI.txt",
        "POLVERARIGI"
    )
    print(result2)

    print("\nStoring more embeddings for user1 (auto-detected data source)...")
    result3 = store_user_embedding(
        "2024-06-JUN-OTHER_DOCUMENTS-POLVERARIGI.txt",
        "POLVERARIGI"
    )
    print(result3)

    print("\nStoring more embeddings for user1 (auto-detected data source)...")
    result4 = store_user_embedding(
        "2024-07-IRS_LETTERS-IRS_LETTER-POLVERARIGI.txt",
        "POLVERARIGI"
    )
    print(result4)

    print("\nStoring more embeddings for user1 (auto-detected data source)...")
    result5 = store_user_embedding(
        "2024-12-DEC-STATEMENTS-POLVERARIGI-CITI_BANK-4221.txt",
        "POLVERARIGI"
    )
    print(result5)

    # Search embeddings for user1
    print("\nSearching user1's embeddings for 'AI'...")
    search_results = search_user_embeddings("Citibank", "POLVERARIGI", k=2)
    print("Search Results:",search_results)