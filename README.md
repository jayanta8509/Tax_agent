# Tax Document Assistant API

An AI-powered tax document assistant built with FastAPI, LangChain, and FAISS that enables users to store, search, and analyze their personal tax documents using RAG (Retrieval-Augmented Generation) technology.

## Features

- 🔍 **RAG-based Document Search**: Intelligent search through your stored documents using embeddings
- 👤 **User-specific Storage**: Each user gets their own isolated document database
- 🤖 **AI-Powered Analysis**: Get comprehensive answers about your tax documents
- 📄 **Multi-format Support**: Process text files containing tax documents
- 🔐 **Memory & Context**: Maintains conversation history for follow-up questions
- 🌐 **REST API**: FastAPI-based service with clean endpoints

## Project Structure

```
Tax_agent/
├── app.py              # FastAPI server with endpoints
├── embedding2.py       # Document embedding and storage functionality
├── rag.py             # RAG system with AI agent
├── embadding.py       # Original embedding implementation
├── .env               # Environment variables (OpenAI API key)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Key Components

### 1. **Document Embedding System** (`embedding2.py`)
- Stores documents in FAISS vector databases
- User-specific storage paths
- Automatic metadata extraction
- Search functionality with similarity scoring

### 2. **RAG System** (`rag.py`)
- LangChain-based AI agent
- Context retrieval from user documents
- Memory system for conversation history
- Intelligent document analysis

### 3. **FastAPI Server** (`app.py`)
- RESTful API endpoints
- User authentication by user_id
- Error handling and logging
- CORS support for web integration

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Tax_agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn
   pip install langchain langchain-openai langchain-community
   pip install python-dotenv faiss-cpu
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

5. **Prepare your documents**
   - Place text files in the project directory
   - Example: `2022-05-MAY-PASSPORT-POLVERARIGI.txt`

## Usage

### 1. **Store Documents**

Using the API:
```python
import requests

response = requests.post("http://localhost:8000/documents/store", json={
    "user_id": "POLVERARIGI",
    "file_path": "passport_document.txt",
    "data_source": "passport"
})
```

Using Python directly:
```python
from embedding2 import store_user_embedding

result = store_user_embedding(
    file_path="passport_document.txt",
    user_id="POLVERARIGI",
    data_source="passport"
)
```

### 2. **Query Your Documents**

API Request:
```python
response = requests.post("http://localhost:8000/chat/agent", json={
    "user_id": "POLVERARIGI",
    "query": "Show me my passport information"
})

answer = response.json()["response"]
print(answer)
```

### 3. **Get User Information**

```bash
curl "http://localhost:8000/users/POLVERARIGI/info"
```

## API Endpoints

### `POST /chat/agent`
Ask questions about your stored documents.

**Request:**
```json
{
    "user_id": "POLVERARIGI",
    "query": "Tell me about my Citibank statements"
}
```

**Response:**
```json
{
    "response": "Based on your documents, I found information about your Citibank statements...",
    "status_code": 200,
    "query": "Tell me about my Citibank statements",
    "user_id": "POLVERARIGI",
    "timestamp": 1704067200.0
}
```

### `POST /documents/store`
Store a new document for a user.

**Request:**
```json
{
    "user_id": "POLVERARIGI",
    "file_path": "/path/to/document.txt",
    "data_source": "tax_document"
}
```

**Response:**
```json
{
    "status": "success",
    "result": {
        "status": "success",
        "document_id": "uuid-here",
        "storage_path": "user_embeddings/POLVERARIGI/POLVERARIGI_faiss_index"
    },
    "user_id": "POLVERARIGI",
    "timestamp": 1704067200.0
}
```

### `GET /users/{user_id}/info`
Get information about a user's stored documents.

**Response:**
```json
{
    "status": "success",
    "user_info": {
        "status": "success",
        "user_id": "POLVERARIGI",
        "total_documents": 5,
        "data_sources": {
            "txt": 3,
            "passport": 1,
            "irs_letter": 1
        },
        "storage_path": "user_embeddings/POLVERARIGI/POLVERARIGI_faiss_index"
    }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "service": "Tax Document Assistant API",
    "version": "1.0.0",
    "features": ["RAG document search", "User-specific storage", "Document management"]
}
```

## Example Document Names

The system works with various document naming conventions:

- `2022-05-MAY-PASSPORT-POLVERARIGI.txt`
- `2022-12-CHECK_IRS-TAX_REFUND_122022-POLVERARIGI.txt`
- `2024-06-JUN-OTHER_DOCUMENTS-POLVERARIGI.txt`
- `2024-07-IRS_LETTERS-IRS_LETTER-POLVERARIGI.txt`
- `2024-12-DEC-STATEMENTS-POLVERARIGI-CITI_BANK-4221.txt`

## Running the Server

1. **Start the API server**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:8000`

2. **Interactive RAG mode**
   ```bash
   python rag.py
   ```

3. **Document processing**
   ```bash
   python embedding2.py
   ```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Directory Structure

- `user_embeddings/`: Stores user-specific FAISS indices
  - `user_embeddings/{user_id}/{user_id}_faiss_index`: Individual user database

## Example Queries

Here are some example queries you can ask about your documents:

- "Show me my passport information"
- "Tell me about my tax refund status"
- "What does my IRS letter say?"
- "Show me my Citibank statement details"
- "What personal documents do I have stored?"
- "Find information about my bank statements"
- "What are my tax document dates?"

## Security Considerations

- User data is stored locally in isolated directories
- No cross-user data access
- API key should be kept secure
- Consider adding authentication for production use

## Development

### Adding New Features

1. **Document Types**: Extend the metadata extraction in `embedding2.py`
2. **Custom Queries**: Modify the prompt in `rag.py`
3. **New Endpoints**: Add routes in `app.py`

### Testing

```bash
# Test basic functionality
python embedding2.py

# Test RAG system
python rag.py

# Start API server
python app.py
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your `.env` file contains a valid OpenAI API key
   - Check that the key has sufficient credits

2. **Document Not Found**
   - Verify file paths are correct
   - Ensure files are readable text documents

3. **No Search Results**
   - Check if documents are stored for the user
   - Try different query terms

4. **Memory Issues**
   - Monitor user embedding storage size
   - Implement document cleanup if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and personal use. Please ensure compliance with OpenAI's terms of service and data privacy regulations.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue in the repository

---

**Built with**: FastAPI, LangChain, FAISS, OpenAI Embeddings
**Purpose**: Intelligent tax document management and analysis