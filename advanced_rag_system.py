"""
Advanced LLM-Powered Intelligent Query-Retrieval System
Enterprise-grade solution for insurance, legal, HR, and compliance domains

Features:
- Multi-format document processing (PDF, DOCX, Email)
- Advanced RAG with LangGraph orchestration
- Semantic search with FAISS/ChromaDB
- Explainable AI decisions
- Real-time clause matching
- Token-optimized processing
"""

import os
import json
import asyncio
import requests
import tempfile
import traceback
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import logging

# Core libraries
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Document processing
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredEmailLoader,
    UnstructuredURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# LLM and orchestration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# LangGraph for advanced workflow
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # LangGraph not available - using standard LangChain workflow (this is fine)

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType:
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx" 
    EMAIL = "email"
    URL = "url"
    
class QueryType:
    """Query type classification"""
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    PROCEDURE = "procedure"
    CONDITION = "condition"
    AMOUNT = "amount"
    TIMELINE = "timeline"

class QueryRequest(BaseModel):
    """Request model for document queries"""
    documents: Union[str, List[str]] = Field(..., description="Document URLs or paths")
    questions: List[str] = Field(..., description="List of questions to answer")
    document_type: Optional[str] = Field(default="auto", description="Document type hint")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing options")

class QueryResponse(BaseModel):
    """Response model for query results"""
    answers: List[str] = Field(..., description="List of answers")
    confidence_scores: List[float] = Field(..., description="Confidence for each answer")
    source_citations: List[List[str]] = Field(..., description="Source citations for each answer")
    processing_time: float = Field(..., description="Total processing time")
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")

class DocumentProcessor:
    """Advanced document processing with multi-format support"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.doc': self._load_docx,
            '.eml': self._load_email,
            '.msg': self._load_email
        }
        
    def detect_document_type(self, document_path: str) -> str:
        """Detect document type from path or URL"""
        if document_path.startswith('http'):
            return DocumentType.URL
        
        suffix = Path(document_path).suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.doc': DocumentType.DOCX,
            '.eml': DocumentType.EMAIL,
            '.msg': DocumentType.EMAIL
        }
        return type_mapping.get(suffix, DocumentType.PDF)
    
    def _load_pdf(self, path: str):
        """Load PDF document with enhanced metadata"""
        loader = PyPDFLoader(path)
        documents = loader.load()
        
        # Enhance metadata for each page
        for i, doc in enumerate(documents):
            doc.metadata.update({
                'source': path,
                'document_type': 'pdf',
                'page_number': i + 1,  # Ensure page numbers start from 1
                'total_pages': len(documents)
            })
        
        return documents
    
    def _load_docx(self, path: str):
        """Load DOCX document with enhanced metadata"""
        loader = Docx2txtLoader(path)
        documents = loader.load()
        
        # Enhance metadata
        for doc in documents:
            doc.metadata.update({
                'source': path,
                'document_type': 'docx',
                'page': 1  # DOCX typically doesn't have clear page breaks
            })
        
        return documents
    
    def _load_email(self, path: str):
        """Load email document"""
        return UnstructuredEmailLoader(path).load()
    
    def _load_url(self, url: str):
        """Load document from URL with smart optimizations maintaining quality"""
        import requests
        import tempfile
        
        try:
            # Smart download with streaming for better memory usage
            response = requests.get(url, timeout=20, stream=True)
            response.raise_for_status()
            
            # Determine file type from content-type or URL
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                suffix = '.pdf'
            elif 'docx' in content_type or url.lower().endswith('.docx'):
                suffix = '.docx'
            elif 'doc' in content_type or url.lower().endswith('.doc'):
                suffix = '.doc'
            else:
                # Default to PDF for unknown types
                suffix = '.pdf'
            
            # Save to temporary file with optimized streaming
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                # Stream download in larger chunks for efficiency
                for chunk in response.iter_content(chunk_size=16384):  # 16KB chunks
                    tmp_file.write(chunk)
                temp_path = tmp_file.name
            
            # Process the downloaded file
            if suffix == '.pdf':
                documents = self._load_pdf(temp_path)
            elif suffix in ['.docx', '.doc']:
                documents = self._load_docx(temp_path)
            else:
                documents = self._load_pdf(temp_path)  # fallback
            
            # Cleanup temporary file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors
                
            return documents
            
        except Exception as e:
            logger.error(f"Error downloading/processing URL {url}: {str(e)}")
            raise
    
    async def process_document(self, document_path: str) -> List[Any]:
        """Process document based on type"""
        try:
            doc_type = self.detect_document_type(document_path)
            
            if doc_type == DocumentType.URL:
                documents = self._load_url(document_path)
            else:
                suffix = Path(document_path).suffix.lower()
                if suffix in self.supported_formats:
                    documents = self.supported_formats[suffix](document_path)
                else:
                    raise ValueError(f"Unsupported document format: {suffix}")
            
            logger.info(f"Successfully processed {doc_type} document: {len(documents)} pages")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

class AdvancedEmbeddingManager:
    """Optimized embedding management with caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = None
        self._cache = {}
        
    def initialize(self):
        """Initialize embedding model with speed optimization"""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    'device': 'cpu'
                },
                encode_kwargs={
                    'batch_size': 32,  # Process in batches for speed
                    'normalize_embeddings': True  # Faster similarity computation
                }
            )
            logger.info(f"Initialized embedding model: {self.model_name}")
        return self.embeddings

class IntelligentChunker:
    """Intelligent document chunking with context preservation"""
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def create_chunks(self, documents: List[Any]) -> List[Any]:
        """Create intelligent chunks with metadata preservation"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Document sections
                "\n\n",    # Paragraphs
                "\n",      # Lines  
                ". ",      # Sentences
                " ",       # Words
                ""
            ],
            keep_separator=True  # Keep separators for context
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Enhanced metadata for quality while preserving original metadata
        for i, chunk in enumerate(chunks):
            # Preserve original metadata (page, source, etc.)
            original_metadata = chunk.metadata.copy()
            
            # Add enhanced metadata without overwriting original
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_size': len(chunk.page_content),
                'created_at': datetime.now().isoformat(),
                # Preserve original metadata
                **original_metadata
            })
            
            # Ensure source is set if not present
            if 'source' not in chunk.metadata or not chunk.metadata['source']:
                # Extract filename from document source
                for doc in documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        chunk.metadata['source'] = doc.metadata['source']
                        break
            
        logger.info(f"Created {len(chunks)} intelligent chunks with preserved metadata")
        return chunks

class VectorStoreManager:
    """Advanced vector store management with FAISS and ChromaDB support"""
    
    def __init__(self, store_type: str = "faiss", persist_directory: str = "./vector_store"):
        self.store_type = store_type.lower()
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.vector_store = None
        
    def create_vector_store(self, chunks: List[Any], embeddings) -> Any:
        """Create optimized vector store"""
        try:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                # Save FAISS index
                self.vector_store.save_local(str(self.persist_directory / "faiss_index"))
                
            elif self.store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=str(self.persist_directory / "chroma_db")
                )
                
            logger.info(f"Created {self.store_type} vector store with {len(chunks)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vector_store(self, embeddings) -> Optional[Any]:
        """Load existing vector store"""
        try:
            if self.store_type == "faiss":
                index_path = self.persist_directory / "faiss_index"
                if index_path.exists():
                    # Check if the existing index is compatible with current embeddings
                    try:
                        self.vector_store = FAISS.load_local(
                            str(index_path), 
                            embeddings,
                            allow_dangerous_deserialization=True
                        )
                        logger.info(f"Loaded existing {self.store_type} vector store")
                    except AssertionError as e:
                        logger.warning(f"FAISS dimension mismatch - recreating vector store: {str(e)}")
                        # Remove the incompatible index
                        import shutil
                        if index_path.exists():
                            shutil.rmtree(str(index_path))
                        return None
                    
            elif self.store_type == "chroma":
                db_path = self.persist_directory / "chroma_db"
                if db_path.exists():
                    self.vector_store = Chroma(
                        persist_directory=str(db_path),
                        embedding_function=embeddings
                    )
                    logger.info(f"Loaded existing {self.store_type} vector store")
                    
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

class QueryClassifier:
    """Intelligent query classification for optimized processing"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.COVERAGE: [
                "cover", "coverage", "include", "benefit", "eligible",
                "reimburse", "pay", "compensate"
            ],
            QueryType.EXCLUSION: [
                "exclude", "exclusion", "not cover", "except", "limitation",
                "restrict", "prohibit", "bar"
            ],
            QueryType.PROCEDURE: [
                "procedure", "surgery", "treatment", "operation", "therapy",
                "intervention", "process"
            ],
            QueryType.CONDITION: [
                "condition", "requirement", "criteria", "prerequisite",
                "qualify", "eligible", "must", "should"
            ],
            QueryType.AMOUNT: [
                "amount", "cost", "price", "fee", "charge", "limit",
                "maximum", "minimum", "sum", "value"
            ],
            QueryType.TIMELINE: [
                "when", "time", "period", "duration", "deadline", "date",
                "waiting", "grace", "term"
            ]
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type for optimized processing"""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                scores[query_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return QueryType.COVERAGE  # Default

class AdvancedRAGSystem:
    """Advanced RAG system with smart caching for speed without quality compromise"""
    
    def __init__(self, 
                 model_name: str = "gemini-1.5-flash",
                 vector_store_type: str = "faiss",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        self.vector_store_type = vector_store_type
        self.embedding_model = embedding_model
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = AdvancedEmbeddingManager(embedding_model)
        self.chunker = IntelligentChunker()
        self.vector_store_manager = VectorStoreManager(vector_store_type)
        self.query_classifier = QueryClassifier()
        
        # LLM setup
        self.llm = None
        # Initialize simple memory store instead of deprecated ConversationBufferMemory
        self.chat_history = []
        
        # Smart caching for speed without quality loss
        self.vector_store_cache = {}
        self.query_cache = {}
        
        # Statistics
        self.token_usage = {"input_tokens": 0, "output_tokens": 0}
        
    def initialize_llm(self):
        """Initialize LLM with balanced speed and quality settings"""
        if self.llm is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            print(f"🔑 Using API Key: {api_key[:20]}...")
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Fast but high-quality model
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=1024,  # Lower output tokens for faster response
                top_p=0.8,
                top_k=40,  # Keep quality parameters
                timeout=60,  # Reasonable timeout
                max_retries=2  # Allow retries for reliability
            )
            logger.info(f"Initialized LLM: {self.model_name}")
        return self.llm
    
    async def process_documents(self, document_paths: Union[str, List[str]]) -> Any:
        """Process multiple documents efficiently with smart caching and async download"""
        if isinstance(document_paths, str):
            document_paths = [document_paths]

        # Create cache key for document set
        cache_key = "|".join(sorted(document_paths))

        # Check if we already processed these documents
        if cache_key in self.vector_store_cache:
            logger.info("Using cached vector store for faster processing")
            return self.vector_store_cache[cache_key]

        async def process_one(doc_path):
            try:
                # Always await the async process_document method
                documents = await self.document_processor.process_document(doc_path)
                return documents
            except Exception as e:
                logger.error(f"Failed to process {doc_path}: {str(e)}")
                return []

        # Run all document processing tasks concurrently
        tasks = [asyncio.create_task(process_one(doc_path)) for doc_path in document_paths]
        results = await asyncio.gather(*tasks)

        # Flatten results and filter out failures
        all_documents = []
        for docs in results:
            if docs:
                all_documents.extend(docs)

        if not all_documents:
            raise ValueError("No documents could be processed successfully")

        # Create intelligent chunks
        chunks = self.chunker.create_chunks(all_documents)

        # Initialize embeddings
        embeddings = self.embedding_manager.initialize()

        # Try to load existing vector store, create if not found
        vector_store = self.vector_store_manager.load_vector_store(embeddings)
        if vector_store is None:
            vector_store = self.vector_store_manager.create_vector_store(chunks, embeddings)

        # Cache the vector store for future use
        self.vector_store_cache[cache_key] = vector_store

        return vector_store
    
    def create_speed_optimized_prompt(self, query_type: str, context: str, question: str) -> str:
        """Create speed-optimized prompts for faster responses"""
        
        base_instructions = """You are an expert document analyst. Answer directly and concisely based on the provided context."""
        
        type_specific_instructions = {
            QueryType.COVERAGE: "Focus on coverage details.",
            QueryType.EXCLUSION: "Focus on exclusions and limitations.", 
            QueryType.PROCEDURE: "Focus on procedures and processes.",
            QueryType.CONDITION: "Focus on conditions and requirements.",
            QueryType.AMOUNT: "Focus on amounts and limits.",
            QueryType.TIMELINE: "Focus on timelines and deadlines."
        }
        
        specific_instruction = type_specific_instructions.get(query_type, "")
        
        # Shortened prompt for speed
        prompt = f"""{base_instructions} {specific_instruction}

Context: {context[:2000]}

Question: {question}

Answer concisely with key points and confidence level:"""
        
        return prompt

    def create_optimized_prompt(self, query_type: str, context: str, question: str) -> str:
        """Create optimized prompts based on query type"""
        base_instructions = """You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. 
Answer the question directly in 3-5 sentences maximum. Only include information relevant to the question. For every key point, provide a citation (document and section/page). Do not list all policy details—focus only on the specific question."""
        type_specific_instructions = {
            QueryType.COVERAGE: "Focus on what IS covered, benefits, inclusions, and eligibility criteria.",
            QueryType.EXCLUSION: "Focus on what IS NOT covered, limitations, restrictions, and exclusions.",
            QueryType.PROCEDURE: "Focus on step-by-step processes, requirements, and procedures.",
            QueryType.CONDITION: "Focus on conditions, requirements, criteria, and qualifications.",
            QueryType.AMOUNT: "Focus on monetary amounts, limits, costs, and financial details.",
            QueryType.TIMELINE: "Focus on timeframes, deadlines, waiting periods, and temporal aspects."
        }
        specific_instruction = type_specific_instructions.get(query_type, "")
        prompt = f"""{base_instructions}

{specific_instruction}

**CRITICAL REQUIREMENTS:**
1. Use ONLY the provided context to answer
2. Answer in 3-5 sentences maximum
3. For every key point, provide a citation (document and section/page)
4. If information is insufficient, clearly state what's missing
5. Be precise and avoid speculation

**Context:**
{context}

**Question:** {question}

**Instructions for Response:**
- Answer directly and concisely (3-5 sentences)
- Provide supporting details only for the specific question
- Reference chunks by their page numbers (e.g., "as stated on Page 9" or "according to Page 33")
- End with confidence level (High/Medium/Low)

**Answer:**"""
        return prompt
    
    async def answer_question(self, question: str, vector_store: Any, top_k: int = 5) -> Dict[str, Any]:
        """Answer a single question with balanced speed and quality"""
        start_time = datetime.now()
        step_times = {}
        
        try:
            # Step 1: Classify query type
            t0 = datetime.now()
            query_type = self.query_classifier.classify_query(question)
            step_times['classify_query'] = (datetime.now() - t0).total_seconds()

            # Step 2: Retrieve relevant documents
            t1 = datetime.now()
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            relevant_docs = retriever.invoke(question)
            step_times['retrieve_docs'] = (datetime.now() - t1).total_seconds()

            # Step 3: Prepare context and citations with improved accuracy
            t2 = datetime.now()
            context_parts = []
            citations = []
            
            # Debug: Log metadata for troubleshooting
            logger.info(f"Retrieved {len(relevant_docs)} documents for citation processing")
            
            for i, doc in enumerate(relevant_docs[:2]):
                # Debug: Log available metadata
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
                
                # Get page information more accurately
                page = doc.metadata.get('page', None)
                if page is None:
                    # Try alternative page keys
                    page = doc.metadata.get('page_number', 
                           doc.metadata.get('page_label', 'N/A'))
                
                # Handle page 0 (often cover page or metadata issue)
                if page == 0:
                    page = "Cover Page"
                
                # Get source information
                source = doc.metadata.get('source', 'Unknown')
                if source == 'Unknown' or not source:
                    # Try to extract filename if source is missing
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    source = f"Document_{chunk_id}"
                
                # Clean up source path to show just filename
                if source != 'Unknown' and '\\' in source:
                    source = source.split('\\')[-1]  # Get just the filename
                elif source != 'Unknown' and '/' in source:
                    source = source.split('/')[-1]   # Handle forward slashes too
                
                # Get section/paragraph information from content for better user understanding
                section_info = ""
                content_preview = doc.page_content[:300].strip()  # First 300 chars
                
                # Try to detect meaningful sections, definitions, or clauses
                import re
                # Look for numbered sections, definitions, or headings
                section_patterns = [
                    r'^(\d+\.?\s*[A-Za-z][^:\n]{5,50}):',  # "36. Policy Schedule:"
                    r'^([A-Z][A-Z\s]{10,40}):',            # "POLICY SCHEDULE:"
                    r'^\s*([A-Z][a-z\s]{10,40})\s*[-:]',   # "Coverage Details -"
                ]
                
                for pattern in section_patterns:
                    section_match = re.search(pattern, content_preview)
                    if section_match:
                        section_info = section_match.group(1).strip()
                        break
                
                content = doc.page_content
                if len(content) > 1200:
                    content = content[:1200] + "..."
                    
                # Just provide the content without confusing chunk labels
                context_parts.append(content)
                
                # Create user-friendly citation with meaningful section if found
                citation_parts = [f"Source: {source}"]
                if str(page) != 'N/A' and page is not None:
                    citation_parts.append(f"Page: {page}")
                if section_info:
                    citation_parts.append(f"Section: {section_info}")
                    
                final_citation = ", ".join(citation_parts)
                citations.append(final_citation)
                
                # Debug: Log final citation
                logger.info(f"Generated citation: {final_citation}")
                
            context = "\n\n".join(context_parts)
            step_times['prepare_context'] = (datetime.now() - t2).total_seconds()

            # Step 4: Prompt creation
            t3 = datetime.now()
            prompt = self.create_optimized_prompt(query_type, context, question)
            step_times['create_prompt'] = (datetime.now() - t3).total_seconds()

            # Step 5: LLM response
            t4 = datetime.now()
            llm = self.initialize_llm()
            try:
                response = llm.invoke(prompt)
                answer = response.content
            except Exception as llm_error:
                logger.error(f"LLM invoke error: {type(llm_error).__name__}: {str(llm_error)}")
                raise llm_error
            step_times['llm_invoke'] = (datetime.now() - t4).total_seconds()

            # Step 6: Confidence extraction
            t5 = datetime.now()
            confidence = self._extract_confidence(answer)
            step_times['extract_confidence'] = (datetime.now() - t5).total_seconds()

            # Step 7: Token usage update
            t6 = datetime.now()
            input_tokens = len(prompt.split())
            output_tokens = len(answer.split())
            self.token_usage["input_tokens"] += input_tokens
            self.token_usage["output_tokens"] += output_tokens
            step_times['update_tokens'] = (datetime.now() - t6).total_seconds()

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"Step timings: {step_times}")

            return {
                "answer": answer,
                "confidence": confidence,
                "citations": citations[:2],  # Limit to top 2 citations for speed
                "processing_time": processing_time,
                "step_times": step_times,
                "query_type": query_type,
                "context_chunks": len(relevant_docs)
            }
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else f"{type(e).__name__} (no message)"
            logger.error(f"Error answering question: {error_msg}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "answer": f"Error processing question: {error_msg}",
                "confidence": 0.0,
                "citations": [],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "query_type": "unknown",
                "context_chunks": 0
            }
    
    def _extract_confidence(self, answer: str) -> float:
        """Extract confidence score from answer"""
        answer_lower = answer.lower()
        if "high confidence" in answer_lower or "certain" in answer_lower:
            return 0.9
        elif "medium confidence" in answer_lower or "likely" in answer_lower:
            return 0.7
        elif "low confidence" in answer_lower or "uncertain" in answer_lower:
            return 0.5
        else:
            return 0.8  # Default confidence
    
    async def process_query_request(self, request: QueryRequest, top_k: int = 10) -> QueryResponse:
        """Process complete query request with parallel processing for speed"""
        start_time = datetime.now()
        
        try:
            # Process documents
            vector_store = await self.process_documents(request.documents)
            
            # Process questions in parallel for speed
            import asyncio
            tasks = []
            for question in request.questions:
                task = asyncio.create_task(self.answer_question(question, vector_store, top_k=top_k))
                tasks.append(task)
            
            # Wait for all questions to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing question {i}: {str(result)}")
                    final_results.append({
                        "answer": f"Error processing question: {str(result)}",
                        "confidence": 0.0,
                        "citations": [],
                        "processing_time": 0.0,
                        "query_type": "unknown",
                        "context_chunks": 0
                    })
                else:
                    final_results.append(result)
            
            # Compile response
            total_time = (datetime.now() - start_time).total_seconds()
            
            response = QueryResponse(
                answers=[r["answer"] for r in final_results],
                confidence_scores=[r["confidence"] for r in final_results],
                source_citations=[r["citations"] for r in final_results],
                processing_time=total_time,
                token_usage=self.token_usage.copy()
            )
            
            logger.info(f"Processed {len(request.questions)} questions in {total_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query request: {str(e)}")
            raise

# Global instance
rag_system = AdvancedRAGSystem()

if __name__ == "__main__":
    # Example usage
    print("🚀 Advanced LLM-Powered Intelligent Query-Retrieval System")
    print("=" * 60)
    print("Features:")
    print("✅ Multi-format document processing")
    print("✅ Advanced RAG with semantic search")
    print("✅ Intelligent query classification")
    print("✅ Optimized token usage")
    print("✅ Explainable AI decisions")
    print("✅ Real-time processing")
    print("=" * 60)