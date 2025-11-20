import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.stores import FAISSVectorStore

def main():
    print("Bootstrapping index...")
    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss")
    
    if not os.path.exists(index_dir):
        print(f"Creating directory: {index_dir}")
        os.makedirs(index_dir, exist_ok=True)
    else:
        print(f"Directory exists: {index_dir}")

    # Initialize store (this might download models if in FAISS mode)
    try:
        store = FAISSVectorStore(index_dir=index_dir)
        print("VectorStore initialized successfully.")
    except Exception as e:
        print(f"Error initializing VectorStore: {e}")
        sys.exit(1)

    print("Bootstrap complete.")

if __name__ == "__main__":
    main()
