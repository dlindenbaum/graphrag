# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for Gemini-specific configuration validation in LanguageModelConfig."""

import pytest
from pydantic import ValidationError

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType, AuthType
from graphrag.config.errors import (
    ApiKeyMissingError,
    ConflictingSettingsError,
)

# Valid base config for Gemini Chat
VALID_GEMINI_CHAT_CONFIG_MINIMAL = {
    "type": ModelType.GeminiChat,
    "model": "gemini-pro",
    "api_key": "test_gemini_api_key",
}

# Valid base config for Gemini Embedding
VALID_GEMINI_EMBEDDING_CONFIG_MINIMAL = {
    "type": ModelType.GeminiEmbedding,
    "model": "embedding-001",
    "api_key": "test_gemini_api_key",
}

def test_valid_gemini_chat_config():
    """Test that a minimal valid Gemini Chat config passes validation."""
    try:
        LanguageModelConfig(**VALID_GEMINI_CHAT_CONFIG_MINIMAL)
    except ValidationError as e:
        pytest.fail(f"Minimal valid Gemini Chat config failed validation: {e}")

def test_valid_gemini_embedding_config():
    """Test that a minimal valid Gemini Embedding config passes validation."""
    try:
        LanguageModelConfig(**VALID_GEMINI_EMBEDDING_CONFIG_MINIMAL)
    except ValidationError as e:
        pytest.fail(f"Minimal valid Gemini Embedding config failed validation: {e}")

def test_gemini_config_with_project_and_location():
    """Test Gemini config with optional project_id and location."""
    config_data = {
        **VALID_GEMINI_CHAT_CONFIG_MINIMAL,
        "gemini_project_id": "my-gcp-project",
        "gemini_location": "us-central1",
    }
    try:
        LanguageModelConfig(**config_data)
    except ValidationError as e:
        pytest.fail(f"Gemini config with project and location failed: {e}")

def test_gemini_missing_api_key():
    """Test that Gemini config raises ApiKeyMissingError if api_key is missing."""
    config_data = {**VALID_GEMINI_CHAT_CONFIG_MINIMAL}
    del config_data["api_key"]
    with pytest.raises(ApiKeyMissingError, match="API key is missing for model type gemini_chat and auth type api_key"):
        LanguageModelConfig(**config_data)

def test_gemini_empty_api_key():
    """Test that Gemini config raises ApiKeyMissingError if api_key is empty."""
    config_data = {**VALID_GEMINI_CHAT_CONFIG_MINIMAL, "api_key": "  "}
    with pytest.raises(ApiKeyMissingError, match="API key is missing for model type gemini_chat and auth type api_key"):
        LanguageModelConfig(**config_data)

def test_gemini_with_azure_managed_identity():
    """Test that Gemini config raises ConflictingSettingsError with AzureManagedIdentity."""
    config_data = {
        **VALID_GEMINI_CHAT_CONFIG_MINIMAL,
        "auth_type": AuthType.AzureManagedIdentity,
    }
    with pytest.raises(ConflictingSettingsError, match="auth_type of azure_managed_identity is not supported for model type gemini_chat"):
        LanguageModelConfig(**config_data)

@pytest.mark.parametrize(
    "azure_field, field_value",
    [
        ("api_base", "https://azure.openai.com"),
        ("api_version", "2023-05-15"),
        ("deployment_name", "my-azure-deployment"),
    ],
)
def test_gemini_with_conflicting_azure_fields(azure_field, field_value):
    """Test Gemini config raises ConflictingSettingsError with Azure-specific fields."""
    config_data = {
        **VALID_GEMINI_CHAT_CONFIG_MINIMAL,
        azure_field: field_value,
    }
    # The error message in the code is "Azure-specific fields (api_base, api_version, deployment_name) should not be set for Gemini models."
    # It does not list the specific field that caused the error.
    with pytest.raises(ConflictingSettingsError, match="Azure-specific fields .* should not be set for Gemini models."):
        LanguageModelConfig(**config_data)

def test_gemini_with_conflicting_openai_organization():
    """Test Gemini config raises ConflictingSettingsError with OpenAI 'organization' field."""
    config_data = {
        **VALID_GEMINI_CHAT_CONFIG_MINIMAL,
        "organization": "org-12345",
    }
    # This assumes 'organization' is caught by _validate_gemini_settings.
    with pytest.raises(ConflictingSettingsError, match="OpenAI-specific field \\(organization\\) should not be set for Gemini models."):
        LanguageModelConfig(**config_data)

def test_azure_config_still_valid():
    """Test that a valid Azure OpenAI config still passes validation (regression check)."""
    azure_config = {
        "type": ModelType.AzureOpenAIChat,
        "model": "gpt-35-turbo",
        "api_key": "test_azure_api_key",
        "api_base": "https://myaccount.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "deployment_name": "my-deployment",
        "auth_type": AuthType.APIKey,
    }
    try:
        LanguageModelConfig(**azure_config)
    except ValidationError as e:
        pytest.fail(f"Valid Azure config failed validation: {e}")

def test_openai_config_still_valid():
    """Test that a valid OpenAI config still passes validation (regression check)."""
    openai_config = {
        "type": ModelType.OpenAIChat,
        "model": "gpt-3.5-turbo",
        "api_key": "test_openai_api_key",
    }
    try:
        LanguageModelConfig(**openai_config)
    except ValidationError as e:
        pytest.fail(f"Valid OpenAI config failed validation: {e}")
