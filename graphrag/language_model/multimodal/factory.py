# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Factory for creating Multimodal Language Model instances."""

import logging

from graphrag.config.enums import ModelType
from graphrag.config.models import LanguageModelConfig
from .base import MultimodalLLM
from .openai_llm import OpenAIMultimodalLLM
from .vertexai_llm import VertexAIMultimodalLLM

log = logging.getLogger(__name__)

class MultimodalLLMFactory:
    """Factory class to create instances of MultimodalLLM."""

    @staticmethod
    def create(config: LanguageModelConfig) -> MultimodalLLM:
        """
        Create a MultimodalLLM instance based on the provided configuration.

        Parameters
        ----------
        config : LanguageModelConfig
            The language model configuration.

        Returns
        -------
        MultimodalLLM
            An instance of a MultimodalLLM subclass.

        Raises
        ------
        ValueError
            If the model type in the configuration is not supported for multimodal operations.
        """
        model_type = config.type

        log.info(f"Creating MultimodalLLM for type: {model_type} with model: {config.model}")

        if model_type == ModelType.OpenAIMultimodal or            model_type == ModelType.AzureOpenAIMultimodal:
            return OpenAIMultimodalLLM(config)

        if model_type == ModelType.GeminiMultimodal:
            return VertexAIMultimodalLLM(config)

        # Potentially add other providers here in the future

        msg = f"Unsupported multimodal model type: {model_type}. Please check your configuration."
        log.error(msg)
        raise ValueError(msg)
