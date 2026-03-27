import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class VectorKnowledgeIndexer:
    def __init__(self, data_path: str = "../public/knowledge.json", index_path: str = "faiss_index.bin"):
        self.data_path = os.path.abspath(data_path)
        self.index_path = index_path
        # Extremely fast CPU-friendly transformer model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.index = None

    def build_index(self):
        print(f"Scanning localized knowledge base at {self.data_path}...")
        if not os.path.exists(self.data_path):
            print("ERROR: knowledge.json not found at target path.")
            return

        with open(self.data_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        # Ingest text strings
        self.chunks = [item['content'] for item in data if 'content' in item]
        print(f"Acquired {len(self.chunks)} semantic chunks for processing.")

        if not self.chunks:
            return

        print("Executing mathematical semantic embeddings matrix...")
        embeddings = self.encoder.encode(self.chunks)
        
        # Build FAISS Flat L2 index for CPU performance
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))

        faiss.write_index(self.index, self.index_path)
        
        # Output mapping for retrieval inference decoder
        with open("chunk_mapping.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)
            
        print(f"SUCCESS: FAISS Euclidean vector space mapped and exported to {self.index_path}")

    def search(self, query: str, top_k: int = 3) -> list:
        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                with open("chunk_mapping.json", "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            else:
                return ["Knowledge base vectors absent. Awaiting indexing."]

        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(np.array(query_vector, dtype=np.float32), top_k)
        
        results = [self.chunks[idx] for idx in indices[0] if idx in range(len(self.chunks))]
        return results

if __name__ == "__main__":
    indexer = VectorKnowledgeIndexer()
    indexer.build_index()
