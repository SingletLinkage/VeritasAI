"""
Vector Store Manager for Local Evidence Storage
Stores pre-scraped fact-checking data and common claims
"""

from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import json
from datetime import datetime

from langchain_community.vectorstores import FAISS
try:
    # Try new import first (recommended)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fall back to old import if new package not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')
from langchain.docstore.document import Document
from dotenv import load_dotenv

from backend.retrieval_models import Evidence, generate_evidence_id

load_dotenv()


class VectorStoreManager:
    """Manages local FAISS vector store for evidence retrieval"""
    
    def __init__(self, store_path: str = "./data/vector_store"):
        """
        Initialize vector store manager
        
        Args:
            store_path: Path to store FAISS index
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Use HuggingFace embeddings (local, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        self._load_or_create_store()
    
    def _load_or_create_store(self):
        """Load existing vector store or create new one"""
        index_path = self.store_path / "faiss_index"
        
        if index_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded existing vector store from {self.store_path}")
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("Creating new vector store...")
                self._create_new_store()
        else:
            self._create_new_store()
    
    def _create_new_store(self):
        """Create new vector store with initial seed data"""
        # Seed data with common fact-check claims
        seed_documents = self._get_seed_data()
        
        if seed_documents:
            self.vector_store = FAISS.from_documents(
                seed_documents,
                self.embeddings
            )
            self.save_store()
            print(f"✅ Created new vector store with {len(seed_documents)} seed documents")
        else:
            # Create empty store
            dummy_doc = Document(
                page_content="Initialization document",
                metadata={"source": "system", "type": "init"}
            )
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            print("✅ Created empty vector store")
    
    def _get_seed_data(self) -> List[Document]:
        """Load seed data from JSON file or return default claims"""
        seed_file = self.store_path / "seed_data.json"
        
        if seed_file.exists():
            with open(seed_file, 'r') as f:
                data = json.load(f)
                return [
                    Document(
                        page_content=item['content'],
                        metadata=item.get('metadata', {})
                    )
                    for item in data
                ]
        
        # Default seed data - common fact-check scenarios
        default_claims = [
            {
                "content": "COVID-19 vaccines contain microchips for tracking. FALSE - No evidence supports this claim. Vaccines contain biological material, not electronic components.",
                "metadata": {
                    "source_url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
                    "source_name": "WHO Myth Busters",
                    "verdict": "FALSE",
                    "category": "health",
                    "date": "2021-01-15"
                }
            },
            {
                "content": "5G networks cause coronavirus spread. FALSE - Viruses cannot travel on radio waves. COVID-19 is spreading in countries without 5G networks.",
                "metadata": {
                    "source_url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
                    "source_name": "WHO Myth Busters",
                    "verdict": "FALSE",
                    "category": "health",
                    "date": "2020-04-01"
                }
            },
            {
                "content": "Climate change is a natural cycle. MISLEADING - While Earth has natural climate cycles, current warming is primarily caused by human activities. Scientific consensus attributes 95%+ of recent warming to human causes.",
                "metadata": {
                    "source_url": "https://climate.nasa.gov/evidence/",
                    "source_name": "NASA Climate Evidence",
                    "verdict": "MISLEADING",
                    "category": "climate",
                    "date": "2023-03-20"
                }
            },
            {
                "content": "Drinking hot water kills coronavirus. FALSE - There is no evidence that drinking hot water prevents or treats COVID-19. The virus can only be killed inside the body by the immune system.",
                "metadata": {
                    "source_url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
                    "source_name": "WHO Myth Busters",
                    "verdict": "FALSE",
                    "category": "health",
                    "date": "2020-03-15"
                }
            },
            {
                "content": "The Eiffel Tower was completed in 1889 for the World's Fair. TRUE - The Eiffel Tower was built for the 1889 Exposition Universelle (World's Fair) held in Paris to celebrate the 100th anniversary of the French Revolution.",
                "metadata": {
                    "source_url": "https://www.toureiffel.paris/en/the-monument/history",
                    "source_name": "Official Eiffel Tower Website",
                    "verdict": "TRUE",
                    "category": "history",
                    "date": "2024-01-10"
                }
            },
            {
                "content": "Vaccines cause autism. FALSE - Multiple large-scale studies have found no link between vaccines and autism. The original study claiming this was retracted due to fraudulent data.",
                "metadata": {
                    "source_url": "https://www.cdc.gov/vaccinesafety/concerns/autism.html",
                    "source_name": "CDC Vaccine Safety",
                    "verdict": "FALSE",
                    "category": "health",
                    "date": "2023-06-12"
                }
            },
            {
                "content": "The Great Wall of China is visible from space. FALSE - The Great Wall is not visible from low Earth orbit with the naked eye. Astronauts have confirmed this misconception is false.",
                "metadata": {
                    "source_url": "https://www.nasa.gov/vision/space/workinginspace/great_wall.html",
                    "source_name": "NASA",
                    "verdict": "FALSE",
                    "category": "science",
                    "date": "2022-09-05"
                }
            },
            {
                "content": "Humans only use 10% of their brain. FALSE - Brain imaging shows that humans use virtually every part of the brain, and most of the brain is active almost all the time.",
                "metadata": {
                    "source_url": "https://www.scientificamerican.com/article/do-people-only-use-10-percent-of-their-brains/",
                    "source_name": "Scientific American",
                    "verdict": "FALSE",
                    "category": "science",
                    "date": "2023-02-18"
                }
            }
        ]
        
        return [
            Document(
                page_content=claim['content'],
                metadata=claim['metadata']
            )
            for claim in default_claims
        ]
    
    def search(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Search vector store for relevant evidence
        
        Args:
            query: Search query (claim to verify)
            top_k: Number of results to return
            
        Returns:
            List of Evidence objects
        """
        if not self.vector_store:
            print("⚠️ Vector store not initialized")
            return []
        
        try:
            # Search with scores
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k
            )
            
            evidences = []
            for doc, score in results:
                # Convert distance to similarity score (lower distance = higher similarity)
                # FAISS returns L2 distance, so we need to convert it
                similarity_score = 1 / (1 + score)  # Normalize to 0-1 range
                
                evidence = Evidence(
                    id=generate_evidence_id(
                        doc.page_content,
                        doc.metadata.get('source_url', 'local_store')
                    ),
                    content=doc.page_content,
                    source_url=doc.metadata.get('source_url', 'local_vector_store'),
                    source_name=doc.metadata.get('source_name', 'Local Knowledge Base'),
                    retrieval_score=min(similarity_score, 1.0),  # Cap at 1.0
                    published_date=doc.metadata.get('date'),
                    snippet=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    metadata={
                        **doc.metadata,
                        "verdict": doc.metadata.get('verdict'),
                        "category": doc.metadata.get('category')
                    },
                    retrieval_method="vector_store"
                )
                evidences.append(evidence)
            
            return evidences
            
        except Exception as e:
            print(f"❌ Vector store search failed: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to vector store"""
        if not self.vector_store:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.save_store()
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def save_store(self):
        """Save vector store to disk"""
        try:
            self.vector_store.save_local(str(self.store_path))
            print(f"✅ Saved vector store to {self.store_path}")
        except Exception as e:
            print(f"❌ Failed to save vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "index_path": str(self.store_path),
            "total_documents": self.vector_store.index.ntotal
        }


# Singleton instance
_vector_store_manager = None

def get_vector_store_manager() -> VectorStoreManager:
    """Get or create vector store manager singleton"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
