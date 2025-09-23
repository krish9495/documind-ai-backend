# Documind-AI Backend

## 🚀 Advanced LLM-Powered Intelligent Query-Retrieval System

Enterprise-grade solution for insurance, legal, HR, and compliance document analysis.

---

## Features

- Multi-format document processing (PDF, DOCX, Email, URL)
- Advanced Retrieval-Augmented Generation (RAG) with semantic search
- State-of-the-art citation system: page, section, clause detection
- Configurable chunking, retrieval, and caching
- Batch processing endpoint for multi-question workflows
- Explainable AI decisions with confidence scoring
- Real-time clause matching and metadata preservation
- Production-ready FastAPI backend

---

## Quick Start

1. **Clone the repository:**
   ```sh
   git clone https://github.com/krish9495/documind-ai-backend.git
   cd documind-ai-backend
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file with your Google API key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```
4. **Run the server:**
   ```sh
   python api_server.py
   ```
5. **Access API docs:**
   - Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive documentation

---

## API Endpoints

### `/api/v1/batch` (POST)

Batch process multiple documents and questions with configurable options.

#### Example Request

```json
{
  "documents": "path/to/document.pdf",
  "questions": ["What is the sum insured?", "What are the exclusions?"],
  "document_format": "auto",
  "processing_options": {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "top_k_retrieval": 10,
    "include_metadata": true,
    "optimize_for_speed": false,
    "enable_caching": true
  },
  "session_id": "test-session-001"
}
```

#### Example Response

```json
{
  "session_id": "test-session-001",
  "results": [
    {
      "question": "What is the sum insured?",
      "answer": "The sum insured is specified in the Policy Schedule...",
      "confidence_score": 0.9,
      "source_citations": [
        "Source: policy.pdf, Page: 4, Section: Clause 3.2 Sum Insured Details"
      ],
      "processing_time": 2.1,
      "context_chunks_used": 8
    }
  ],
  "processing_summary": {
    "total_questions": 2,
    "documents_processed": 1,
    "average_confidence": 0.9,
    "chunk_size_used": 500
  }
}
```

---

## Configuration Options

- **chunk_size**: Size of each text chunk (100-2000)
- **chunk_overlap**: Overlap between chunks (0-500)
- **top_k_retrieval**: Number of top chunks to retrieve (1-20)
- **include_metadata**: Include source info in results
- **optimize_for_speed**: Fast vs accurate processing
- **enable_caching**: Cache results for repeated queries

---

## Authentication

- All endpoints require a Bearer token in the `Authorization` header.
- Example:
  ```http
  Authorization: Bearer hackrx-2024-bajaj-finserv
  ```

---

## Citation System

- Citations include document name, page number, and detected section/clause.
- Example: `Source: policy.pdf, Page: 4, Section: Clause 3.2 Sum Insured Details`
- Supports numbered, roman numeral, and lettered clauses for precise cross-referencing.

---

## License

MIT

---

## Contact

For questions or support, contact [krish9495](https://github.com/krish9495).
