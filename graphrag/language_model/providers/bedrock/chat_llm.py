import json
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.protocol import ChatModel
from graphrag.language_model.response.base import ModelResponse

logger = logging.getLogger(__name__)


class BedrockChatLLM(ChatModel):
    """A ChatModel implementation that uses AWS Bedrock."""

    def __init__(self, config: LanguageModelConfig):
        """Initialize the BedrockChatLLM."""
        self.config = config
        # Consider AWS SDK retry strategy vs. GraphRAG's built-in retry
        # For now, let's use a basic boto3 client configuration
        boto_config = BotoConfig(
            region_name=config.aws_region,
            retries={
                "max_attempts": config.max_retries or 0,  # boto default is 5 if not set
                # "mode": "standard" # or "adaptive"
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

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response for the given text."""
        # This is a simplified example for Anthropic Claude.
        # Real implementation needs to handle different Bedrock models.
        # Also, history needs to be formatted according to the model's requirements.
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": self.config.max_tokens or 2000,  # Bedrock default, ensure it uses config
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            # Add other Anthropic-specific parameters as needed
        })

        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.config.model,  # e.g., "anthropic.claude-v2"
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            completion = response_body.get("completion")

            # Construct ModelResponse
            # Usage information might vary by Bedrock model provider
            # For Claude: response_body often contains some form of token count or stop reason
            # This is a placeholder for actual usage parsing
            raw_response = response_body
            output_text = completion or ""

            # Placeholder for token counts, actual calculation depends on model
            prompt_tokens = len(prompt) // 4  # Rough estimate
            completion_tokens = len(output_text) // 4  # Rough estimate
            total_tokens = prompt_tokens + completion_tokens

            return ModelResponse(
                output_text=output_text,
                raw_response=raw_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error("Error invoking Bedrock model: %s", e)
            # Consider how to handle errors, maybe re-raise or return an error response
            raise

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response for the given text asynchronously."""
        # For now, this will be a wrapper around the synchronous version.
        # Proper async implementation would use aioboto3 or similar.
        # This is a common pattern in the existing codebase for providers that don't have native async support.
        return self.chat(prompt, history, **kwargs)

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        """Generate a response for the given text using a streaming interface."""
        # This is a simplified example for Anthropic Claude.
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": self.config.max_tokens or 2000,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        })

        try:
            response = self.client.invoke_model_with_response_stream(
                body=body,
                modelId=self.config.model,
                accept="application/json",
                contentType="application/json",
            )

            stream = response.get("body")
            if stream:
                for event in stream:
                    chunk = event.get("chunk")
                    if chunk:
                        chunk_obj = json.loads(chunk.get("bytes").decode())
                        if "completion" in chunk_obj:
                            yield chunk_obj["completion"]
                        elif "delta" in chunk_obj and "text" in chunk_obj["delta"]:  # For Cohere style streaming
                            yield chunk_obj["delta"]["text"]
                        # Add handling for other model stream formats (e.g. Meta Llama)
        except Exception as e:
            logger.error("Error streaming from Bedrock model: %s", e)
            raise

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a response for the given text using an async streaming interface."""
        # Similar to achat, this would ideally use an async library.
        # For now, wraps the synchronous generator.
        # This is NOT truly async and will block.
        # Proper async streaming with Bedrock requires more setup (e.g. aiobotocore)
        # and handling of async iteration over the stream.
        # This is a placeholder to fulfill the protocol.
        for chunk in self.chat_stream(prompt, history, **kwargs):
            yield chunk
