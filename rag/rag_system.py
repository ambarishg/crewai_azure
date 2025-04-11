from llm_helper.interface_llm_helper import ILLMHelper
import qdrant_client as qc
import qdrant_client.http.models as qmodels
from qdrant_client.http.models import *
from sentence_transformers import SentenceTransformer
import os
import logging

class RAGSystem:
    def __init__(self, generator: ILLMHelper):
        self.generator = generator
        # Initialize Qdrant components
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading model: %s", os.getenv("MODEL_NAME"))
        
    
    def retrieve(self, query: str) -> str:
        self.model = SentenceTransformer(os.getenv("MODEL_NAME"))
        self.client = qc.QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_KEY")
        )
        vector = self.model.encode(query, convert_to_tensor=True).tolist()
        results = self.client.search(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            query_vector=vector,
            limit=5
        )
        return "\n---\n".join([r.payload["content"] for r in results])
    
    def query(self, query: str) -> str:
        context = self.retrieve(query)
        return self.generator.generate(context, query)
