import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.language_model.providers.bedrock.chat_llm import BedrockChatLLM
from graphrag.language_model.response.base import ModelResponse

class TestBedrockChatLLM(unittest.TestCase):
    def _create_config(self, model="anthropic.claude-v2"):
        return LanguageModelConfig(
            type=ModelType.BedrockChat,
            model=model,
            aws_region="us-west-2",
            aws_access_key_id="fake_id",
            aws_secret_access_key="fake_secret",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

    @patch("graphrag.language_model.providers.bedrock.chat_llm.boto3.client")
    def test_chat_success(self, mock_boto_client):
        mock_llm_response = {
            "completion": "This is a test response.",
            # Add other fields Bedrock might return for token counts if applicable
        }
        mock_response_stream = BytesIO(json.dumps(mock_llm_response).encode("utf-8"))

        mock_bedrock_runtime = MagicMock()
        mock_bedrock_runtime.invoke_model.return_value = {
            "body": mock_response_stream,
            "contentType": "application/json",
        }
        mock_boto_client.return_value = mock_bedrock_runtime

        config = self._create_config()
        llm = BedrockChatLLM(config)
        prompt = "Test prompt"
        response = llm.chat(prompt)

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.output_text, "This is a test response.")
        # self.assertEqual(response.prompt_tokens, ...) # Add assertions if token counts are parsed
        # self.assertEqual(response.completion_tokens, ...)
        mock_bedrock_runtime.invoke_model.assert_called_once()
        called_args, called_kwargs = mock_bedrock_runtime.invoke_model.call_args
        self.assertEqual(called_kwargs.get("modelId"), "anthropic.claude-v2")
        self.assertIn(prompt, json.loads(called_kwargs.get("body")).get("prompt"))

    @patch("graphrag.language_model.providers.bedrock.chat_llm.boto3.client")
    def test_chat_stream_success(self, mock_boto_client):
        mock_chunk1 = {"chunk": {"bytes": json.dumps({"completion": "Hello "}).encode("utf-8")}}
        mock_chunk2 = {"chunk": {"bytes": json.dumps({"completion": "World!"}).encode("utf-8")}}

        mock_bedrock_runtime = MagicMock()
        mock_bedrock_runtime.invoke_model_with_response_stream.return_value = {
            "body": [mock_chunk1, mock_chunk2] # Simulate iterable stream of events
        }
        mock_boto_client.return_value = mock_bedrock_runtime

        config = self._create_config()
        llm = BedrockChatLLM(config)
        prompt = "Test stream prompt"

        response_parts = list(llm.chat_stream(prompt))

        self.assertEqual(len(response_parts), 2)
        self.assertEqual(response_parts[0], "Hello ")
        self.assertEqual(response_parts[1], "World!")
        mock_bedrock_runtime.invoke_model_with_response_stream.assert_called_once()
        called_args, called_kwargs = mock_bedrock_runtime.invoke_model_with_response_stream.call_args
        self.assertEqual(called_kwargs.get("modelId"), "anthropic.claude-v2")

    # TODO: Add tests for achat and achat_stream if/when they become truly async
    # TODO: Add tests for error handling (e.g., Bedrock API errors)
    # TODO: Add tests for different model providers if logic is added (e.g. Cohere, AI21)
