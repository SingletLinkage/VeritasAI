"""
Vector Store Population Script
Loads scraped fact-checking data from JSON files and populates the vector store
"""

from typing import List, Dict
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.docstore.document import Document
from backend.vector_store import get_vector_store_manager
from datetime import datetime


class VectorStorePopulator:
    """Populates vector store with scraped fact-checking data"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "web_scrappers"
        self.json_files = {
            "who_data.json": "WHO (World Health Organization)",
            "factcheck_data.json": "FactCheck.org",
            "pti_data.json": "PTI Fact Check",
            "rbi_data.json": "Reserve Bank of India"
        }
    
    def load_json_file(self, filepath: Path) -> List[Dict]:
        """Load data from a JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error in {filepath}: {e}")
            return []
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return []
    
    def convert_to_documents(self, data: List[Dict], source_name: str) -> List[Document]:
        """Convert JSON data to LangChain Documents"""
        documents = []
        
        for item in data:
            try:
                # Extract content
                content = item.get('content', '').strip()
                if not content:
                    continue
                
                # Extract metadata
                metadata = item.get('metadata', {})
                
                # Ensure all required fields exist
                if not metadata.get('source_name'):
                    metadata['source_name'] = source_name
                
                if not metadata.get('source_url'):
                    metadata['source_url'] = 'Unknown'
                
                # Normalize verdict to uppercase for consistency
                if 'verdict' in metadata:
                    verdict = metadata['verdict']
                    if isinstance(verdict, str):
                        metadata['verdict'] = verdict.upper()
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"⚠️  Error processing item: {e}")
                continue
        
        return documents
    
    def load_all_data(self) -> List[Document]:
        """Load all scraped data from JSON files"""
        all_documents = []
        
        print("\n📂 Loading scraped data from JSON files...")
        print("-" * 70)
        
        for filename, source_name in self.json_files.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"⚠️  Skipping {filename} (not found)")
                continue
            
            print(f"\n📄 Loading: {filename}")
            print(f"   Source: {source_name}")
            
            # Load JSON data
            data = self.load_json_file(filepath)
            
            if not data:
                print(f"   ⚠️  No data found in {filename}")
                continue
            
            print(f"   ✅ Loaded {len(data)} entries")
            
            # Convert to documents
            documents = self.convert_to_documents(data, source_name)
            print(f"   ✅ Converted to {len(documents)} documents")
            
            all_documents.extend(documents)
        
        return all_documents
    
    def populate_vector_store(self, documents: List[Document]):
        """Populate vector store with documents"""
        if not documents:
            print("\n❌ No documents to add to vector store")
            return
        
        print("\n📚 Initializing vector store...")
        vector_store = get_vector_store_manager()
        
        print(f"\n💾 Adding {len(documents)} documents to vector store...")
        vector_store.add_documents(documents)
        
        # Show stats
        stats = vector_store.get_stats()
        print(f"\n✅ Vector store populated successfully!")
        print(f"   Total documents: {stats.get('total_documents', len(documents))}")
        print(f"   Index path: {stats.get('index_path', 'data/vector_store')}")
    
    def test_search(self):
        """Test vector store search functionality"""
        print("\n🔍 Testing search functionality...")
        print("-" * 70)
        
        vector_store = get_vector_store_manager()
        
        test_queries = [
            "COVID-19 vaccine safety",
            "financial fraud RBI",
            "fake news video",
            "health misinformation"
        ]
        
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            results = vector_store.search(query, top_k=3)
            
            if not results:
                print("    No results found")
                continue
            
            for idx, result in enumerate(results, 1):
                print(f"    {idx}. {result.source_name}")
                print(f"       Score: {result.retrieval_score:.3f}")
                print(f"       Category: {result.metadata.get('category', 'N/A')}")
                snippet = result.content[:100].replace('\n', ' ')
                print(f"       Snippet: {snippet}...")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Populate vector store with fact-checking data from scraped JSON files"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test search functionality without repopulating"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip search testing after population"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VECTOR STORE POPULATION - SCRAPED DATA")
    print("=" * 70)
    
    populator = VectorStorePopulator()
    
    if args.test_only:
        # Only test search
        print("\n🧪 Running search tests only...")
        populator.test_search()
    else:
        # Load and populate
        documents = populator.load_all_data()
        
        if documents:
            print("\n" + "=" * 70)
            print(f"📊 SUMMARY: Loaded {len(documents)} total documents")
            print("=" * 70)
            
            # Count by source
            source_counts = {}
            for doc in documents:
                source = doc.metadata.get('source_name', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print("\n📈 Documents by source:")
            for source, count in sorted(source_counts.items()):
                print(f"   • {source}: {count} documents")
            
            # Count by category
            category_counts = {}
            for doc in documents:
                category = doc.metadata.get('category', 'uncategorized')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            print("\n📋 Documents by category:")
            for category, count in sorted(category_counts.items()):
                print(f"   • {category}: {count} documents")
            
            # Populate vector store
            populator.populate_vector_store(documents)
            
            # Test search unless disabled
            if not args.no_test:
                populator.test_search()
        else:
            print("\n❌ No documents loaded. Check JSON files in web_scrappers/")
    
    print("\n" + "=" * 70)
    print("✅ VECTOR STORE READY FOR USE")
    print("=" * 70)
    print("\nUsage in code:")
    print("  from backend.vector_store import get_vector_store_manager")
    print("  vs = get_vector_store_manager()")
    print("  results = vs.search('your query', top_k=5)")
    print()


if __name__ == "__main__":
    main()
