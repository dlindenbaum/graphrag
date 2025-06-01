# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Base class for Multimodal Language Models."""

from abc import ABC, abstractmethod
from typing import Any

from graphrag.config.models import LanguageModelConfig


class MultimodalLLM(ABC):
    """Abstract base class for Multimodal Language Models."""

    def __init__(self, config: LanguageModelConfig):
        """
        Initialize the MultimodalLLM.

        Parameters
        ----------
        config : LanguageModelConfig
            The configuration for the language model.
        """
        self.config = config

    @abstractmethod
    async def describe_image(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Describe an image using the multimodal LLM.

        Parameters
        ----------
        url : str
            The URL of the image to describe.
        prompt_override : str | None, optional
            An optional prompt to override the default image description prompt
            from the configuration. If None, the default prompt will be used.
        kwargs : Any
            Additional keyword arguments for the underlying LLM API.

        Returns
        -------
        str
            A textual description of the image.
        """
        pass

    @abstractmethod
    async def summarize_video(
        self,
        url: str,
        prompt_override: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Summarize a video using the multimodal LLM.

        Parameters
        ----------
        url : str
            The URL of the video to summarize.
        prompt_override : str | None, optional
            An optional prompt to override the default video summarization prompt
            from the configuration. If None, the default prompt will be used.
        kwargs : Any
            Additional keyword arguments for the underlying LLM API.

        Returns
        -------
        str
            A textual summary of the video.
        """
        pass

    def _get_prompt(self, task_key: str, prompt_override: str | None) -> str:
        """Helper to get the prompt for a given task."""
        if prompt_override:
            return prompt_override

        default_prompts = self.config.multimodal_prompts or {}
        prompt = default_prompts.get(task_key)

        if not prompt:
            # Fallback to a very generic prompt if not configured
            # This should ideally be configured by the user for good results
            if task_key == "image_description":
                return "Describe this image in detail."
            if task_key == "video_summary":
                return "Summarize this video."
            return f"Process this {task_key.replace('_', ' ')}." # Should not happen if called correctly

        return prompt
