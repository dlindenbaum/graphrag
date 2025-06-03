# Using AWS Bedrock with GraphRAG

GraphRAG supports using AWS Bedrock as a backend for language models (both chat and embeddings). This allows you to leverage a variety of foundation models available through the Bedrock service.

## Configuration

To use AWS Bedrock, you need to configure it in your `settings.yaml` file or through environment variables. Here's an example of how to configure a Bedrock chat model and an embedding model:

```yaml
models:
  bedrock_chat_model: # A custom name for your model configuration
    type: bedrock_chat
    model: "anthropic.claude-v2" # Or any other Bedrock chat model ID, e.g., "meta.llama2-70b-chat-v1"
    api_key: ${GRAPHRAG_AWS_ACCESS_KEY_ID} # Can also be set directly
    aws_secret_access_key: ${GRAPHRAG_AWS_SECRET_ACCESS_KEY} # Can also be set directly
    aws_session_token: ${GRAPHRAG_AWS_SESSION_TOKEN} # Optional, for temporary credentials
    aws_region: "us-east-1" # Your AWS region for Bedrock
    # Optional: Add other LLM parameters like temperature, max_tokens, etc.
    # temperature: 0.7
    # max_tokens: 500

  bedrock_embedding_model: # A custom name for your embedding model
    type: bedrock_embedding
    model: "amazon.titan-embed-text-v1" # Or any other Bedrock embedding model ID, e.g., "cohere.embed-english-v3"
    api_key: ${GRAPHRAG_AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${GRAPHRAG_AWS_SECRET_ACCESS_KEY}
    aws_session_token: ${GRAPHRAG_AWS_SESSION_TOKEN} # Optional
    aws_region: "us-west-2"
```

### Parameters

*   `type`: Must be `bedrock_chat` for chat models or `bedrock_embedding` for embedding models.
*   `model`: The specific model ID for the Bedrock model you want to use (e.g., `anthropic.claude-v2`, `amazon.titan-embed-text-v1`, `cohere.embed-english-v3`, `meta.llama2-70b-chat-v1`).
*   `api_key`: Your AWS Access Key ID. It's recommended to use environment variables (e.g., `GRAPHRAG_AWS_ACCESS_KEY_ID` or `AWS_ACCESS_KEY_ID`).
*   `aws_secret_access_key`: Your AWS Secret Access Key. It's recommended to use environment variables (e.g., `GRAPHRAG_AWS_SECRET_ACCESS_KEY` or `AWS_SECRET_ACCESS_KEY`).
*   `aws_session_token`: Your AWS Session Token (if using temporary credentials). Recommended to use environment variables (e.g., `GRAPHRAG_AWS_SESSION_TOKEN` or `AWS_SESSION_TOKEN`). This field is optional.
*   `aws_region`: The AWS region where your Bedrock models are hosted (e.g., `us-east-1`, `us-west-2`). This is a **required** field for Bedrock.
*   Other standard LLM parameters like `temperature`, `max_tokens`, `top_p`, etc., can also be included and will be passed to the Bedrock model if applicable to that model.

## Authentication

The Bedrock integration uses the provided AWS credentials (access key ID, secret access key, and optional session token) to authenticate with AWS services. Ensure the IAM user or role associated with these credentials has the necessary permissions to invoke the specified Bedrock models (e.g., `bedrock:InvokeModel`).

If `api_key` (for AWS Access Key ID) or `aws_secret_access_key` are not provided directly in the configuration or through their `GRAPHRAG_` prefixed environment variables, the underlying `boto3` library will attempt to find credentials using its standard chain (e.g., environment variables like `AWS_ACCESS_KEY_ID`, shared credential files, IAM roles for EC2/ECS, etc.). However, explicitly providing them or using the `GRAPHRAG_` prefixed environment variables is recommended for clarity within GraphRAG's configuration.

## Model IDs

You can find a list of available model IDs in the AWS Bedrock console or by using the AWS CLI (`aws bedrock list-foundation-models`). Ensure you use the correct model ID for the type of model (chat vs. embedding) and the provider (Anthropic, AI21, Cohere, Amazon, Meta, etc.).

## Using Bedrock Models in Workflows

Once configured in the `models` section, you can reference your Bedrock model configurations in your GraphRAG workflows just like any other model:

```yaml
extract_graph:
  model_id: bedrock_chat_model # Reference the key from the models block
  # ... other extract_graph settings

create_final_text_units:
  text_embedding:
    model_id: bedrock_embedding_model # Reference the key for embedding
    # ... other text_embedding settings
```
