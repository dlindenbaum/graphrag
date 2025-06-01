# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Vertex AI (Gemini) Multimodal LLM implementation."""

import logging
from typing import Any

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold

from graphrag.config.models import LanguageModelConfig
from .base import MultimodalLLM

log = logging.getLogger(__name__)

# Default safety settings - can be made configurable if needed
DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

class VertexAIMultimodalLLM(MultimodalLLM):
    """Vertex AI (Gemini) Multimodal LLM implementation."""

    def __init__(self, config: LanguageModelConfig):
        super().__init__(config)

        # Initialize Vertex AI. This is usually done once per application session.
        # Subsequent calls don't re-initialize if project and location are the same.
        try:
            vertexai.init(
                project=self.config.gemini_project_id,
                location=self.config.gemini_location,
                # Credentials can be handled by GOOGLE_APPLICATION_CREDENTIALS env var
                # or other ADC mechanisms. If api_key is provided in config,
                # it might be used by specific client libraries if they support it directly,
                # but Vertex AI Python SDK typically relies on ADC.
            )
            log.info(f"Vertex AI initialized for project {self.config.gemini_project_id} and location {self.config.gemini_location}")
        except Exception as e:
            log.error(f"Error initializing Vertex AI: {e}", exc_info=True)
            # Depending on policy, either raise or allow to proceed if client can be created later
            raise

        self.model = GenerativeModel(
            self.config.model, # e.g., "gemini-1.5-pro-latest" or "gemini-pro-vision"
            # system_instruction=[Optional system prompt here if needed]
        )
        log.info(f"Vertex AI GenerativeModel {self.config.model} loaded.")


    async def _generate_content_with_media(
        self,
        media_url: str,
        media_mime_type: str, # e.g., "image/png", "video/mp4"
        prompt_text: str,
        **kwargs: Any
    ) -> str:
        """Helper function to generate content from media URI and prompt."""
        media_part = Part.from_uri(media_url, mime_type=media_mime_type)
        text_part = Part.from_text(prompt_text)

        contents = [text_part, media_part]

        # GenerationConfig mapping from LanguageModelConfig
        # Max output tokens, temperature, top_p, etc.
        generation_config = GenerationConfig(
            max_output_tokens=kwargs.get("max_output_tokens", self.config.max_tokens or 2048), # Gemini default is often 2048 or 8192
            temperature=kwargs.get("temperature", self.config.temperature or 0.7),
            top_p=kwargs.get("top_p", self.config.top_p or 1.0),
            # candidate_count: kwargs.get("n", self.config.n or 1), # Gemini's n
            # stop_sequences: Optional
        )

        # Safety settings - consider making these configurable
        safety_settings = kwargs.get("safety_settings", DEFAULT_SAFETY_SETTINGS)

        try:
            response = await self.model.generate_content_async(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=False # Non-streaming for now
            )

            if response.candidates and response.candidates[0].content.parts:
                # Assuming the first part of the first candidate is the desired text
                # Gemini can return multiple parts (e.g. text, function calls)
                # We are interested in the text part for description/summary
                text_response = ""
                for part in response.candidates[0].content.parts:
                    if part.text:
                        text_response += part.text

                if text_response:
                    return text_response.strip()
                else:
                    log.warning(f"Vertex AI model {self.config.model} returned no text content for {media_url}. Response: {response}")
                    return "" # Or raise an error
            else:
                # Log refusal or empty response
                reason = response.candidates[0].finish_reason if response.candidates else "Unknown"
                safety_ratings_str = str(response.candidates[0].safety_ratings) if response.candidates and response.candidates[0].safety_ratings else "N/A"
                log.error(
                    f"Vertex AI model {self.config.model} returned no content or was blocked for {media_url}. "
                    f"Finish Reason: {reason}. Safety Ratings: {safety_ratings_str}. Prompt: '{prompt_text[:100]}...'"
                )
                # Construct a more informative error message
                error_message = f"No content generated. Finish Reason: {reason}."
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_message += f" Prompt Blocked: {response.prompt_feedback.block_reason_message}."
                return error_message # Return the error/block reason

        except Exception as e:
            log.error(f"Error in Vertex AI _generate_content_with_media for {media_url} with model {self.config.model}: {e}", exc_info=True)
            raise


    async def describe_image(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """Describe an image using the Vertex AI (Gemini) multimodal LLM."""
        prompt_text = self._get_prompt("image_description", prompt_override)

        # Determine MIME type. This is a simplification.
        # A more robust solution might involve checking file headers or using a library.
        # For GCS URLs, MIME type is often part of metadata.
        # Common image types:
        mime_type = "image/png" # Default, can be improved
        if ".jpg" in url.lower() or ".jpeg" in url.lower():
            mime_type = "image/jpeg"
        elif ".webp" in url.lower():
            mime_type = "image/webp"
        # Add more MIME types as needed

        log.info(f"Describing image {url} with mime_type {mime_type} using Vertex AI model {self.config.model}")
        return await self._generate_content_with_media(url, mime_type, prompt_text, **kwargs)

    async def summarize_video(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """Summarize a video using the Vertex AI (Gemini) multimodal LLM.
        Assumes the URL is a GCS URI accessible by the Vertex AI service.
        """
        prompt_text = self._get_prompt("video_summary", prompt_override)

        # Determine MIME type for video.
        # Common video types:
        mime_type = "video/mp4" # Default
        if ".mov" in url.lower():
            mime_type = "video/quicktime" # or video/mp4 if it's h264
        elif ".avi" in url.lower():
            mime_type = "video/x-msvideo"
        elif ".webm" in url.lower():
            mime_type = "video/webm"
        # Add more as needed

        log.info(f"Summarizing video {url} with mime_type {mime_type} using Vertex AI model {self.config.model}")
        return await self._generate_content_with_media(url, mime_type, prompt_text, **kwargs)
