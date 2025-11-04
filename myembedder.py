from typing import List
from langchain_core.embeddings import Embeddings
import httpx
import numpy as np
from loguru import logger

class MyEmbedder(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        try:
            url = "http://101.99.3.94:8080/embed"
            res = httpx.post(url, json={"inputs":texts})
            res.raise_for_status()
            res = res.json()
            return res
        except Exception as e:
            logger.error(f"Error: {e}")
            return []


    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        try:
            url = "http://101.99.3.94:8080/embed"
            res = httpx.post(url, json={"inputs":text})
            res.raise_for_status()
            res = res.json()
            outp = res[0]
            return outp
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

if __name__ == '__main__':
    emb = MyEmbedder()
    print(emb.embed_documents(['he']))