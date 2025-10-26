# CitRAG System Flow Diagram

```
📄 INPUT DOCUMENTS                    🔧 PROCESSING PIPELINE                    📋 OUTPUT
┌─────────────────────┐              ┌─────────────────────────────────────┐   ┌──────────────────┐
│  • PDF Files       │              │                                     │   │                  │
│  • DOCX Documents  │────────────► │  1️⃣ DOCUMENT PROCESSOR              │   │  📝 ANSWER       │
│  • Email Messages  │              │     ├─ Extract text & metadata      │   │     +            │
│  • Web Content     │              │     ├─ Preserve page boundaries     │   │  🔗 CITATIONS    │
└─────────────────────┘              │     └─ Detect document structure    │   │     +            │
                                     │              ⬇️                     │   │  📊 CONFIDENCE   │
                                     │  2️⃣ INTELLIGENT CHUNKER             │   │                  │
┌─────────────────────┐              │     ├─ Structure-aware splitting    │   └──────────────────┘
│  ❓ USER QUERY      │              │     ├─ Overlap for continuity       │
│  "What are the     │              │     └─ Preserve metadata chain      │
│   coverage         │              │              ⬇️                     │
│   exclusions?"     │──────────────┤  3️⃣ VECTOR STORE MANAGER           │
└─────────────────────┘              │     ├─ Generate embeddings          │
                                     │     ├─ FAISS similarity search      │
                                     │     └─ Cache invalidation           │
                                     │              ⬇️                     │
                                     │  4️⃣ CITATION GENERATOR              │
                                     │     ├─ Multi-pattern detection      │
                                     │     ├─ Section identification       │
                                     │     └─ Actionable references        │
                                     │              ⬇️                     │
                                     │  5️⃣ QUERY ORCHESTRATOR             │
                                     │     ├─ LLM answer generation        │
                                     │     ├─ Response assembly            │
                                     │     └─ Batch processing             │
                                     └─────────────────────────────────────┘
```

## 🔄 **Detailed Component Workflow**

### **1️⃣ Document Processor Flow**

```
📄 Raw Document
    ⬇️
├─ Format Detection (PDF/DOCX/Email/Web)
    ⬇️
├─ Text Extraction + Structure Analysis
    ⬇️
├─ Metadata Enrichment
    │  ├─ Source path
    │  ├─ Page numbers
    │  ├─ Document type
    │  ├─ Timestamps
    │  └─ Section headers
    ⬇️
📋 Structured Document Collection
```

### **2️⃣ Intelligent Chunker Flow**

```
📋 Structured Documents
    ⬇️
├─ Recursive Splitting Strategy
    │  ├─ By sections (if detected)
    │  ├─ By paragraphs
    │  └─ By sentences (fallback)
    ⬇️
├─ Overlap Management (200-400 chars)
    ⬇️
├─ Metadata Preservation
    │  ├─ Original source
    │  ├─ Chunk position
    │  ├─ Section context
    │  └─ Boundary info
    ⬇️
🧩 Context-Rich Chunks
```

### **3️⃣ Vector Store Manager Flow**

```
🧩 Context-Rich Chunks
    ⬇️
├─ Embedding Generation
    │  └─ all-MiniLM-L6-v2 (384-dim vectors)
    ⬇️
├─ FAISS Index Creation
    │  └─ Cosine similarity optimization
    ⬇️
├─ Cache Key Generation
    │  └─ hash(file_paths + modification_times)
    ⬇️
🗄️ Searchable Vector Database

❓ User Query
    ⬇️
├─ Query Embedding
    ⬇️
├─ Similarity Search (top-k retrieval)
    ⬇️
📊 Relevant Chunks + Similarity Scores
```

### **4️⃣ Citation Generator Flow**

```
📊 Relevant Chunks
    ⬇️
├─ Multi-Pattern Analysis
    │  ├─ Numbered clauses (1., 2., 3.)
    │  ├─ Roman numerals (i., ii., iii.)
    │  ├─ Lettered sections (a), b), c))
    │  ├─ Hierarchical (1.1, 1.2, 1.3)
    │  └─ Bracketed ((1), (2), (3))
    ⬇️
├─ Section Mapping
    │  ├─ Extract section titles
    │  ├─ Map to document hierarchy
    │  └─ Validate with context
    ⬇️
├─ Citation Assembly
    │  ├─ Source document
    │  ├─ Page number
    │  ├─ Section reference
    │  └─ Confidence score
    ⬇️
🔗 Precise, Actionable Citations
```

### **5️⃣ Query Orchestrator Flow**

```
❓ User Query + 🔗 Citations + 📊 Chunks
    ⬇️
├─ LLM Processing (Gemini-1.5-Flash)
    │  ├─ Context integration
    │  ├─ Answer synthesis
    │  └─ Confidence assessment
    ⬇️
├─ Response Assembly
    │  ├─ Answer text
    │  ├─ Citation formatting
    │  ├─ Confidence scores
    │  └─ Processing metadata
    ⬇️
├─ Quality Validation
    │  ├─ Answer relevance check
    │  ├─ Citation accuracy verify
    │  └─ Response completeness
    ⬇️
📋 Final Response to User
```

## 🎯 **Key Performance Metrics**

```
┌─────────────────────────────────────────────────────────────┐
│  PERFORMANCE DASHBOARD                                      │
├─────────────────────────────────────────────────────────────┤
│  📊 Citation Accuracy:      81%                            │
│  🔗 Actionable Citations:   89%                            │
│  ⚡ Processing Speed:       2.8s avg                       │
│  🎯 System Confidence:     81%                             │
│  📈 Improvement Factor:    2.6x over baseline              │
│  🔄 Batch Scalability:     Linear (28.1s for 10 queries)  │
│  💾 Memory Usage:          Stable across batch sizes       │
└─────────────────────────────────────────────────────────────┘
```

## 🏆 **Competitive Advantage**

```
┌─────────────────────────────────────────────────────────────┐
│  CITRAG vs TRADITIONAL RAG                                 │
├─────────────────────────────────────────────────────────────┤
│                    CitRAG    │    Standard RAG              │
├─────────────────────────────────────────────────────────────┤
│  Citation Format:            │                              │
│  "Policy.pdf, Page 4,        │  "Document 1,                │
│   Section 3.2 Coverage"      │   Chunk 3"                   │
├─────────────────────────────────────────────────────────────┤
│  Actionable Citations:  89%  │  34%                         │
│  Processing Speed:     2.8s  │  4.2s                        │
│  Structure Awareness:   ✅   │  ❌                          │
│  Enterprise Features:   ✅   │  ❌                          │
│  Professional Focus:    ✅   │  ❌                          │
└─────────────────────────────────────────────────────────────┘
```

---

**Use this visual guide during interviews to help explain the system flow and highlight the technical innovations clearly!**
