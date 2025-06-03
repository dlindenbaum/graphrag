import json
import logging
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.protocol import EmbeddingModel

logger = logging.getLogger(__name__)


class BedrockEmbeddingLLM(EmbeddingModel):
    """An EmbeddingModel implementation that uses AWS Bedrock."""

    def __init__(self, config: LanguageModelConfig):
        """Initialize the BedrockEmbeddingLLM."""
        self.config = config
        boto_config = BotoConfig(
            region_name=config.aws_region,
            retries={
                "max_attempts": config.max_retries or 0,
            },
        )
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            aws_session_token=config.aws_session_token,
            region_name=config.aws_region,
            config=boto_config,
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding vector for the given text."""
        # This example is for Amazon Titan Embeddings.
        # Other embedding models on Bedrock (e.g., Cohere) will have different request/response formats.
        if not text.strip():  # Bedrock embedding models can error on empty strings
            logger.warning(
                "Received empty or whitespace-only string for embedding, returning empty list."
            )
            return []

        body = json.dumps({"inputText": text})
        model_id = self.config.model  # e.g., "amazon.titan-embed-text-v1"

        try:
            response = self.client.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embedding = response_body.get("embedding")
            if embedding is None:
                logger.error(
                    f"Bedrock embedding response for model {model_id} did not contain 'embedding' field. Response: {response_body}"
                )
                raise ValueError("Embedding not found in Bedrock response")
            return embedding
        except Exception as e:
            logger.error(f"Error invoking Bedrock embedding model {model_id}: {e}")
            raise

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding vector for the given text asynchronously."""
        # Wrapper around synchronous version for now.
        return self.embed(text, **kwargs)

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""
        results = []
        for text_input in text_list:
            # Bedrock models generally don't support batching in a single API call for embeddings.
            # We have to call them sequentially.
            # Some models might have specific batch endpoints, but `invoke_model` is typically single-item.
            # Consider adding a delay or respecting rate limits if config.requests_per_minute is set.
            results.append(self.embed(text_input, **kwargs))
        return results

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Generate embedding vectors for a list of texts asynchronously."""
        # Wrapper around synchronous version for now.
        # For true async, one might use asyncio.gather with aembed for each item.
        results = []
        for text_input in text_list:
            results.append(await self.aembed(text_input, **kwargs))
        return results
