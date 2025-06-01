# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the Gemini model providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.language_model.providers.gemini.models import GeminiChat, GeminiEmbedding
from graphrag.language_model.response.base import BaseModelResponse # For type hint

# A fixture for default GeminiChat config
@pytest.fixture
def gemini_chat_config():
    return LanguageModelConfig(
        type=ModelType.GeminiChat,
        model="gemini-pro", # Example model
        api_key="test_gemini_api_key",
        # Add other necessary fields like gemini_project_id if your config needs them
    )

# A fixture for default GeminiEmbedding config
@pytest.fixture
def gemini_embedding_config():
    return LanguageModelConfig(
        type=ModelType.GeminiEmbedding,
        model="embedding-001", # Example model
        api_key="test_gemini_api_key",
    )

# ---- GeminiChat Tests ----

@patch("graphrag.language_model.providers.gemini.models.genai.configure") # Mock SDK configure
@patch("graphrag.language_model.providers.gemini.models.genai.GenerativeModel") # Mock SDK model
def test_gemini_chat_initialization(mock_generative_model, mock_configure, gemini_chat_config):
    """Test GeminiChat initialization configures SDK and model."""
    mock_sdk_model_instance = MagicMock()
    mock_generative_model.return_value = mock_sdk_model_instance

    chat_model = GeminiChat(name="test_chat", config=gemini_chat_config)

    # mock_configure.assert_called_once_with(api_key="test_gemini_api_key")
    # mock_generative_model.assert_called_once_with("gemini-pro") # Or whatever model is in config
    assert chat_model.name == "test_chat"
    assert chat_model.config == gemini_chat_config
    # assert chat_model._client == mock_sdk_model_instance # Check if client is the mocked one
    # These asserts are commented out because the actual SDK calls are commented out in the model provider
    # For this phase, we are testing the structure. Actual SDK interaction tests would require more setup.
    assert chat_model._model_name == "gemini-pro"


@pytest.mark.asyncio
async def test_gemini_chat_achat(gemini_chat_config):
    """Test the achat method of GeminiChat (mocked SDK call)."""
    # Since the actual SDK call is mocked/commented out in GeminiChat,
    # this test will currently check the placeholder logic.
    chat_model = GeminiChat(name="test_chat", config=gemini_chat_config)

    # Mock the underlying SDK's async call if it were active
    # For now, we test the current mocked behavior in GeminiChat
    # chat_model._client.generate_content_async = AsyncMock(return_value=MagicMock(text="Expected Gemini Response"))

    prompt = "Hello Gemini!"
    response: BaseModelResponse = await chat_model.achat(prompt)

    assert "Mock response from Gemini for: Hello Gemini!" in response.output.content
    assert response.history is None # As per current mock

@pytest.mark.asyncio
async def test_gemini_chat_achat_stream(gemini_chat_config):
    """Test the achat_stream method of GeminiChat (mocked SDK call)."""
    chat_model = GeminiChat(name="test_chat", config=gemini_chat_config)

    # Mock the underlying SDK's streaming call if it were active
    # async def mock_stream_chunks():
    #     yield MagicMock(text="Chunk1")
    #     yield MagicMock(text="Chunk2")
    # chat_model._client.generate_content_async = AsyncMock(return_value=mock_stream_chunks())

    prompt = "Stream this!"
    full_response = []
    async for chunk in chat_model.achat_stream(prompt):
        full_response.append(chunk)

    assert "".join(full_response) == "Mock stream for: Stream this! part 1, part 2"


# ---- GeminiEmbedding Tests ----

@patch("graphrag.language_model.providers.gemini.models.genai.configure")
@patch("graphrag.language_model.providers.gemini.models.genai.GenerativeModel") # Assuming embedding also uses GenerativeModel or a similar class
def test_gemini_embedding_initialization(mock_generative_model, mock_configure, gemini_embedding_config):
    """Test GeminiEmbedding initialization."""
    mock_sdk_model_instance = MagicMock()
    mock_generative_model.return_value = mock_sdk_model_instance # Or specific embedding model mock

    embedding_model = GeminiEmbedding(name="test_embedding", config=gemini_embedding_config)

    # mock_configure.assert_called_once_with(api_key="test_gemini_api_key")
    # mock_generative_model.assert_called_once_with("embedding-001")
    assert embedding_model.name == "test_embedding"
    assert embedding_model.config == gemini_embedding_config
    # assert embedding_model._client == mock_sdk_model_instance
    assert embedding_model._model_name == "embedding-001"


@pytest.mark.asyncio
async def test_gemini_embedding_aembed(gemini_embedding_config):
    """Test the aembed method of GeminiEmbedding (mocked SDK call)."""
    embedding_model = GeminiEmbedding(name="test_embedding", config=gemini_embedding_config)

    # Mock the underlying SDK's embedding call
    # async def mock_embed_content(model, content, task_type):
    #     return {"embedding": [0.1, 0.2, 0.3]}
    # with patch("graphrag.language_model.providers.gemini.models.genai.embed_content_async", new=AsyncMock(side_effect=mock_embed_content)):
    #     text_to_embed = "Embed this text."
    #     embedding = await embedding_model.aembed(text_to_embed)
    #     assert embedding == [0.1, 0.2, 0.3]

    text_to_embed = "Embed this text."
    embedding = await embedding_model.aembed(text_to_embed)
    assert len(embedding) == 10 # Current mock returns 10 dims
    assert embedding[0] == hash(text_to_embed) / 1e18


@pytest.mark.asyncio
async def test_gemini_embedding_aembed_batch(gemini_embedding_config):
    """Test the aembed_batch method of GeminiEmbedding (mocked SDK call)."""
    embedding_model = GeminiEmbedding(name="test_embedding", config=gemini_embedding_config)

    texts_to_embed = ["text1", "text2"]
    # Mocking strategy similar to aembed if SDK calls were active

    embeddings = await embedding_model.aembed_batch(texts_to_embed)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 10 # Current mock
    assert embeddings[0][0] == hash("text1") / 1e18
    assert embeddings[1][0] == hash("text2") / 1e18

# More tests can be added for:
# - Synchronous wrappers (chat, embed, embed_batch)
# - Error handling (e.g., if API key is missing and SDK would raise error)
# - Cache interaction (if cache object is passed and used)
# - Correct passing of parameters like temperature, max_tokens to the mocked SDK calls
# - History formatting for chat models
