# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Language model configuration."""

from typing import Literal

import tiktoken
from pydantic import BaseModel, Field, model_validator

from graphrag.config.defaults import language_model_defaults
from graphrag.config.enums import AsyncType, AuthType, ModelType
from graphrag.config.errors import (
    ApiKeyMissingError,
    AzureApiBaseMissingError,
    AzureApiVersionMissingError,
    AzureDeploymentNameMissingError,
    ConflictingSettingsError,
)
from graphrag.language_model.factory import ModelFactory


class LanguageModelConfig(BaseModel):
    """Language model configuration."""

    api_key: str | None = Field(
        description="The API key to use for the LLM service.",
        default=language_model_defaults.api_key,
    )
    gemini_project_id: str | None = Field(
        description="The Google Cloud Project ID for Gemini.",
        default=None,
    )
    gemini_location: str | None = Field(
        description="The Google Cloud location for Gemini services.",
        default=None,
    )
    aws_region: str | None = Field(description="The AWS region for Bedrock services.", default=None)
    aws_access_key_id: str | None = Field(description="AWS access key ID for Bedrock.", default=None)
    aws_secret_access_key: str | None = Field(description="AWS secret access key for Bedrock.", default=None)
    aws_session_token: str | None = Field(description="AWS session token for Bedrock (optional).", default=None)
    multimodal_prompts: dict[str, str] | None = Field(
        description="A dictionary of default prompts for multimodal tasks, e.g., {'image_description': 'Describe this image.', 'video_summary': 'Summarize this video.'}",
        default=None,
    )

    def _validate_api_key(self) -> None:
        """Validate the API key.

        API Key is required when using OpenAI API
        or when using Azure API with API Key authentication.
        For the time being, this check is extra verbose for clarity.
        It will also raise an exception if an API Key is provided
        when one is not expected such as the case of using Azure
        Managed Identity.

        Raises
        ------
        ApiKeyMissingError
            If the API key is missing and is required.
        """
        is_gemini = self.type in [
            ModelType.GeminiChat,
            ModelType.GeminiEmbedding,
            ModelType.GeminiMultimodal,
        ]

        if (self.auth_type == AuthType.APIKey or is_gemini) and (
            self.api_key is None or self.api_key.strip() == ""
        ):
            raise ApiKeyMissingError(
                self.type,
                self.auth_type.value if not is_gemini else "api_key",  # Gemini always uses API key
            )

        if self.auth_type == AuthType.AzureManagedIdentity and (
            self.api_key is not None and self.api_key.strip() != ""
        ) and not is_gemini: # Don't raise if it's Gemini, other validator will catch it
            msg = "API Key should not be provided when using Azure Managed Identity. Please rerun `graphrag init` and remove the api_key when using Azure Managed Identity."
            raise ConflictingSettingsError(msg)

    auth_type: AuthType = Field(
        description="The authentication type.",
        default=language_model_defaults.auth_type,
    )

    def _validate_auth_type(self) -> None:
        """Validate the authentication type.

        auth_type must be api_key when using OpenAI and
        can be either api_key or azure_managed_identity when using AOI.

        Raises
        ------
        ConflictingSettingsError
            If the Azure authentication type conflicts with the model being used.
        """
        is_gemini = self.type in [
            ModelType.GeminiChat,
            ModelType.GeminiEmbedding,
            ModelType.GeminiMultimodal,
        ]
        if self.auth_type == AuthType.AzureManagedIdentity and (
            self.type == ModelType.OpenAIChat
            or self.type == ModelType.OpenAIEmbedding
            or self.type == ModelType.OpenAIMultimodal
            or is_gemini
        ):
            msg = f"auth_type of azure_managed_identity is not supported for model type {self.type}. Please rerun `graphrag init` and set the auth_type to api_key (if OpenAI or Gemini) or ensure you are using a compatible model type for Azure Managed Identity."
            raise ConflictingSettingsError(msg)

    type: ModelType | str = Field(description="The type of LLM model to use.")

    def _validate_type(self) -> None:
        """Validate the model type.

        Raises
        ------
        KeyError
            If the model name is not recognized.
        """
        # TODO: Update ModelFactory to officially include multimodal types
        known_multimodal_types = [
            ModelType.OpenAIMultimodal,
            ModelType.AzureOpenAIMultimodal,
            ModelType.GeminiMultimodal,
        ]
        if not ModelFactory.is_supported_model(self.type) and self.type not in known_multimodal_types:
            all_known_types = ModelFactory.get_chat_models() + ModelFactory.get_embedding_models() + known_multimodal_types
            msg = f"Model type {self.type} is not recognized, must be one of {all_known_types}."
            raise KeyError(msg)

    model: str = Field(description="The LLM model to use.")
    encoding_model: str = Field(
        description="The encoding model to use",
        default=language_model_defaults.encoding_model,
    )

    def _validate_encoding_model(self) -> None:
        """Validate the encoding model.

        Raises
        ------
        KeyError
            If the model name is not recognized.
        """
        if self.encoding_model.strip() == "":
            self.encoding_model = tiktoken.encoding_name_for_model(self.model)

    api_base: str | None = Field(
        description="The base URL for the LLM API.",
        default=language_model_defaults.api_base,
    )

    def _validate_api_base(self) -> None:
        """Validate the API base.

        Required when using AOI.

        Raises
        ------
        AzureApiBaseMissingError
            If the API base is missing and is required.
        """
        if (
            self.type == ModelType.AzureOpenAIChat
            or self.type == ModelType.AzureOpenAIEmbedding
        ) and (self.api_base is None or self.api_base.strip() == ""):
            raise AzureApiBaseMissingError(self.type)

    api_version: str | None = Field(
        description="The version of the LLM API to use.",
        default=language_model_defaults.api_version,
    )

    def _validate_api_version(self) -> None:
        """Validate the API version.

        Required when using AOI.

        Raises
        ------
        AzureApiBaseMissingError
            If the API base is missing and is required.
        """
        if (
            self.type == ModelType.AzureOpenAIChat
            or self.type == ModelType.AzureOpenAIEmbedding
        ) and (self.api_version is None or self.api_version.strip() == ""):
            raise AzureApiVersionMissingError(self.type)

    deployment_name: str | None = Field(
        description="The deployment name to use for the LLM service.",
        default=language_model_defaults.deployment_name,
    )

    def _validate_deployment_name(self) -> None:
        """Validate the deployment name.

        Required when using AOI.

        Raises
        ------
        AzureDeploymentNameMissingError
            If the deployment name is missing and is required.
        """
        if (
            self.type == ModelType.AzureOpenAIChat
            or self.type == ModelType.AzureOpenAIEmbedding
        ) and (self.deployment_name is None or self.deployment_name.strip() == ""):
            raise AzureDeploymentNameMissingError(self.type)

    organization: str | None = Field(
        description="The organization to use for the LLM service.",
        default=language_model_defaults.organization,
    )
    proxy: str | None = Field(
        description="The proxy to use for the LLM service.",
        default=language_model_defaults.proxy,
    )
    audience: str | None = Field(
        description="Azure resource URI to use with managed identity for the llm connection.",
        default=language_model_defaults.audience,
    )
    model_supports_json: bool | None = Field(
        description="Whether the model supports JSON output mode.",
        default=language_model_defaults.model_supports_json,
    )
    request_timeout: float = Field(
        description="The request timeout to use.",
        default=language_model_defaults.request_timeout,
    )
    tokens_per_minute: int | Literal["auto"] | None = Field(
        description="The number of tokens per minute to use for the LLM service.",
        default=language_model_defaults.tokens_per_minute,
    )

    def _validate_tokens_per_minute(self) -> None:
        """Validate the tokens per minute.

        Raises
        ------
        ValueError
            If the tokens per minute is less than 0.
        """
        # If the value is a number, check if it is less than 1
        if isinstance(self.tokens_per_minute, int) and self.tokens_per_minute < 1:
            msg = f"Tokens per minute must be a non zero positive number, 'auto' or null. Suggested value: {language_model_defaults.tokens_per_minute}."
            raise ValueError(msg)

    requests_per_minute: int | Literal["auto"] | None = Field(
        description="The number of requests per minute to use for the LLM service.",
        default=language_model_defaults.requests_per_minute,
    )

    def _validate_requests_per_minute(self) -> None:
        """Validate the requests per minute.

        Raises
        ------
        ValueError
            If the requests per minute is less than 0.
        """
        # If the value is a number, check if it is less than 1
        if isinstance(self.requests_per_minute, int) and self.requests_per_minute < 1:
            msg = f"Requests per minute must be a non zero positive number, 'auto' or null. Suggested value: {language_model_defaults.requests_per_minute}."
            raise ValueError(msg)

    retry_strategy: str = Field(
        description="The retry strategy to use for the LLM service.",
        default=language_model_defaults.retry_strategy,
    )
    max_retries: int = Field(
        description="The maximum number of retries to use for the LLM service.",
        default=language_model_defaults.max_retries,
    )

    def _validate_max_retries(self) -> None:
        """Validate the maximum retries.

        Raises
        ------
        ValueError
            If the maximum retries is less than 0.
        """
        if self.max_retries < 1:
            msg = f"Maximum retries must be greater than or equal to 1. Suggested value: {language_model_defaults.max_retries}."
            raise ValueError(msg)

    max_retry_wait: float = Field(
        description="The maximum retry wait to use for the LLM service.",
        default=language_model_defaults.max_retry_wait,
    )
    concurrent_requests: int = Field(
        description="Whether to use concurrent requests for the LLM service.",
        default=language_model_defaults.concurrent_requests,
    )
    async_mode: AsyncType = Field(
        description="The async mode to use.", default=language_model_defaults.async_mode
    )
    responses: list[str | BaseModel] | None = Field(
        default=language_model_defaults.responses,
        description="Static responses to use in mock mode.",
    )
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate.",
        default=language_model_defaults.max_tokens,
    )
    temperature: float = Field(
        description="The temperature to use for token generation.",
        default=language_model_defaults.temperature,
    )
    max_completion_tokens: int | None = Field(
        description="The maximum number of tokens to consume. This includes reasoning tokens for the o* reasoning models.",
        default=language_model_defaults.max_completion_tokens,
    )
    reasoning_effort: str | None = Field(
        description="Level of effort OpenAI reasoning models should expend. Supported options are 'low', 'medium', 'high'; and OAI defaults to 'medium'.",
        default=language_model_defaults.reasoning_effort,
    )
    top_p: float = Field(
        description="The top-p value to use for token generation.",
        default=language_model_defaults.top_p,
    )
    n: int = Field(
        description="The number of completions to generate.",
        default=language_model_defaults.n,
    )
    frequency_penalty: float = Field(
        description="The frequency penalty to use for token generation.",
        default=language_model_defaults.frequency_penalty,
    )
    presence_penalty: float = Field(
        description="The presence penalty to use for token generation.",
        default=language_model_defaults.presence_penalty,
    )

    def _validate_azure_settings(self) -> None:
        """Validate the Azure settings.

        Raises
        ------
        AzureApiBaseMissingError
            If the API base is missing and is required.
        AzureApiVersionMissingError
            If the API version is missing and is required.
        AzureDeploymentNameMissingError
            If the deployment name is missing and is required.
        """
        is_azure_model = self.type in [
            ModelType.AzureOpenAIChat,
            ModelType.AzureOpenAIEmbedding,
            ModelType.AzureOpenAIMultimodal,
        ]
        if is_azure_model:
            self._validate_api_base()
            self._validate_api_version()
            self._validate_deployment_name()

    def _validate_gemini_settings(self) -> None:
        """Validate the Gemini settings.

        Raises
        ------
        ConflictingSettingsError
            If conflicting settings are provided for Gemini models.
        """
        is_gemini_model = self.type in [
            ModelType.GeminiChat,
            ModelType.GeminiEmbedding,
            ModelType.GeminiMultimodal,
        ]
        if is_gemini_model:
            conflicting_azure_fields = [
                self.api_base,
                self.api_version,
                self.deployment_name,
            ]
            if any(field is not None for field in conflicting_azure_fields):
                msg = "Azure-specific fields (api_base, api_version, deployment_name) should not be set for Gemini models."
                raise ConflictingSettingsError(msg)

            # Add checks for OpenAI-specific fields if any become relevant, e.g., self.organization
            # For now, let's assume 'organization' could be a conflicting field if set for Gemini.
            if self.organization is not None:
                msg = "OpenAI-specific field (organization) should not be set for Gemini models."
                raise ConflictingSettingsError(msg)

            # Gemini project_id and location are not strictly required by the library itself for all operations,
            # but could be validated here if they become mandatory for all Gemini use cases.
            # For now, we assume they are optional at this config level.

    def _validate_bedrock_settings(self) -> None:
        """Validate the Bedrock settings.

        Raises
        ------
        ValueError
            If AWS region is missing for Bedrock models.
        ConflictingSettingsError
            If conflicting settings are provided for Bedrock models.
        """
        is_bedrock_model = self.type in [
            ModelType.BedrockChat,
            ModelType.BedrockEmbedding,
        ]
        if is_bedrock_model:
            if not self.aws_region:
                msg = "AWS region is required for Bedrock models."
                raise ValueError(msg)

            # Check for conflicting Azure settings
            conflicting_azure_fields = [
                self.api_base,
                self.api_version,
                self.deployment_name,
            ]
            if any(field is not None for field in conflicting_azure_fields):
                msg = "Azure-specific fields (api_base, api_version, deployment_name) should not be set for Bedrock models."
                raise ConflictingSettingsError(msg)

            # Check for conflicting Gemini settings
            conflicting_gemini_fields = [
                self.gemini_project_id,
                self.gemini_location,
            ]
            if any(field is not None for field in conflicting_gemini_fields):
                msg = "Gemini-specific fields (gemini_project_id, gemini_location) should not be set for Bedrock models."
                raise ConflictingSettingsError(msg)

            # Check for conflicting OpenAI settings (e.g., organization)
            if self.organization is not None:
                msg = "OpenAI-specific field (organization) should not be set for Bedrock models."
                raise ConflictingSettingsError(msg)

    @model_validator(mode="after")
    def _validate_model(self):
        self._validate_type()
        self._validate_auth_type()
        self._validate_api_key()
        self._validate_tokens_per_minute()
        self._validate_requests_per_minute()
        self._validate_max_retries()
        self._validate_azure_settings()
        self._validate_gemini_settings()
        self._validate_bedrock_settings()
        self._validate_encoding_model()
        return self
