import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.language_model.providers.bedrock.embedding_llm import BedrockEmbeddingLLM

class TestBedrockEmbeddingLLM(unittest.TestCase):
    def _create_config(self, model="amazon.titan-embed-text-v1"):
        return LanguageModelConfig(
            type=ModelType.BedrockEmbedding,
            model=model,
            aws_region="us-east-1",
            aws_access_key_id="fake_id",
            aws_secret_access_key="fake_secret",
        )

    @patch("graphrag.language_model.providers.bedrock.embedding_llm.boto3.client")
    def test_embed_success(self, mock_boto_client):
        mock_llm_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_response_stream = BytesIO(json.dumps(mock_llm_response).encode("utf-8"))

        mock_bedrock_runtime = MagicMock()
        mock_bedrock_runtime.invoke_model.return_value = {
            "body": mock_response_stream,
            "contentType": "application/json",
        }
        mock_boto_client.return_value = mock_bedrock_runtime

        config = self._create_config()
        llm = BedrockEmbeddingLLM(config)
        text_input = "Embed this text"
        embedding = llm.embed(text_input)

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        mock_bedrock_runtime.invoke_model.assert_called_once()
        called_args, called_kwargs = mock_bedrock_runtime.invoke_model.call_args
        self.assertEqual(called_kwargs.get("modelId"), "amazon.titan-embed-text-v1")
        self.assertEqual(json.loads(called_kwargs.get("body")).get("inputText"), text_input)

    @patch("graphrag.language_model.providers.bedrock.embedding_llm.boto3.client")
    def test_embed_batch_success(self, mock_boto_client):
        mock_llm_response1 = {"embedding": [0.1, 0.2, 0.3]}
        mock_llm_response2 = {"embedding": [0.4, 0.5, 0.6]}

        mock_bedrock_runtime = MagicMock()
        # Simulate invoke_model being called multiple times with different return values
        mock_bedrock_runtime.invoke_model.side_effect = [
            {
                "body": BytesIO(json.dumps(mock_llm_response1).encode("utf-8")),
                "contentType": "application/json",
            },
            {
                "body": BytesIO(json.dumps(mock_llm_response2).encode("utf-8")),
                "contentType": "application/json",
            },
        ]
        mock_boto_client.return_value = mock_bedrock_runtime

        config = self._create_config()
        llm = BedrockEmbeddingLLM(config)
        text_list = ["Text 1", "Text 2"]
        embeddings = llm.embed_batch(text_list)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        self.assertEqual(mock_bedrock_runtime.invoke_model.call_count, 2)

    def test_embed_empty_string(self):
        config = self._create_config()
        llm = BedrockEmbeddingLLM(config) # boto3.client won't be called here
        embedding = llm.embed("   ") # Whitespace only
        self.assertEqual(embedding, [])

    # TODO: Add tests for aembed and aembed_batch if/when they become truly async
    # TODO: Add tests for error handling (e.g., Bedrock API errors)
