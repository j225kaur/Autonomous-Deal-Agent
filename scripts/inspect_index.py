import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.stores import FAISSVectorStore

def main():
    parser = argparse.ArgumentParser(description="Inspect the vector index.")
    parser.add_argument("--query", type=str, required=True, help="Query string to search for")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss")
    print(f"Loading index from {index_dir}...")
    
    try:
        store = FAISSVectorStore(index_dir=index_dir)
        results = store.search(args.query, k=args.k)
        
        print(f"\nFound {len(results)} results for query '{args.query}':\n")
        for i, doc in enumerate(results):
            print(f"[{i+1}] {doc.page_content[:200]}...")
            print(f"    Metadata: {doc.metadata}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
