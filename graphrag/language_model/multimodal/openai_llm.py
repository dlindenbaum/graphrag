# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Multimodal LLM implementation."""

import logging
from typing import Any
from openai import AsyncOpenAI

from graphrag.config.models import LanguageModelConfig
from .base import MultimodalLLM

log = logging.getLogger(__name__)

class OpenAIMultimodalLLM(MultimodalLLM):
    """OpenAI Multimodal LLM implementation (e.g., for GPT-4o)."""

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)
        # It's good practice to initialize the client here if it's reused
        # The API key and other settings are in self.config
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.api_base, # For Azure OpenAI or custom endpoints
            # Other necessary OpenAI client settings like timeout can be added
            timeout=self.config.request_timeout or 60.0, # Default timeout
            max_retries=self.config.max_retries or 2, # Default retries
        )
        # Note: For Azure, deployment_name is often part of the model string or handled by the client setup.
        # If api_base is set, it assumes an Azure-like endpoint.
        # The actual model name (deployment id for Azure) is in self.config.model

    async def describe_image(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """Describe an image using the OpenAI multimodal LLM."""
        prompt_text = self._get_prompt("image_description", prompt_override)

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,  # e.g., "gpt-4o"
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": url},
                            },
                        ],
                    }
                ],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens or 1024), # Sensible default
                temperature=kwargs.get("temperature", self.config.temperature or 0.7),
                top_p=kwargs.get("top_p", self.config.top_p or 1.0),
                n=kwargs.get("n", self.config.n or 1),
                # other parameters like frequency_penalty, presence_penalty can be added from config
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                log.error("OpenAI describe_image returned no content.")
                return ""
        except Exception as e:
            log.error(f"Error in OpenAI describe_image: {e}", exc_info=True)
            # Consider re-raising or returning a specific error message
            # For now, returning empty string or raising simplifies caller if they expect string
            raise # Re-raise the exception to allow higher-level error handling

    async def summarize_video(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Summarize a video using the OpenAI multimodal LLM.
        NOTE: Direct video URL summarization is not a standard feature of GPT-4o
        without prior frame extraction. This method currently raises NotImplementedError.
        A proper implementation would require a video frame extraction mechanism first.
        """
        log.warning(
            "summarize_video with OpenAI LLM currently requires pre-extracted frames. "
            "Direct video URL summarization is not implemented."
        )
        raise NotImplementedError(
            "Direct video URL summarization with OpenAI LLM requires a frame extraction "
            "and processing pipeline, which is not implemented in this basic version."
        )
