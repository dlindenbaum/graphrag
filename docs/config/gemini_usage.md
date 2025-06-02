# Using Google Gemini Models with GraphRAG

GraphRAG supports the use of Google Gemini models for both chat completion and text embedding tasks. This guide provides instructions on how to configure GraphRAG to use your Gemini models.

## Prerequisites

Before you can use Gemini models with GraphRAG, you will need:

1.  A Google Cloud Project where the Gemini API is enabled.
2.  An API Key associated with your project that has permissions to use the Gemini API.

## Configuration

To use a Gemini model, you need to specify it in your GraphRAG configuration file (typically `settings.yaml` or passed as a dictionary if using GraphRAG as a library). You will define your Gemini models under the `llm` -> `models` section.

### Gemini Chat Model (`gemini_chat`)

For tasks requiring chat completions (e.g., graph extraction, summarization, question answering), you can configure a Gemini chat model.

**Key Configuration Parameters:**

*   `type`: Must be set to `gemini_chat`.
*   `api_key`: Your Google Cloud API key for Gemini. It's highly recommended to use an environment variable (e.g., `${GEMINI_API_KEY}`) for this.
*   `model`: The specific Gemini chat model you want to use (e.g., `"gemini-pro"`, `"gemini-1.0-pro"`, `"gemini-1.5-pro-latest"`).
*   `gemini_project_id` (Optional): Your Google Cloud Project ID. While often not strictly required if the API key is global, it can be good practice to include it.
*   `gemini_location` (Optional): The Google Cloud location/region for your Gemini model if applicable (e.g., `"us-central1"`).
*   Other common parameters like `temperature`, `top_p`, `max_tokens` (which might map to `max_output_tokens` or similar in the Gemini SDK) can also be configured. Refer to the Gemini API documentation for specifics on how these are handled.

**Example YAML Configuration:**

```yaml
llm:
  # ... other llm settings ...
  models:
    my_gemini_chat_model: # You can name your model configuration
      type: gemini_chat
      api_key: ${GEMINI_API_KEY} # Recommended: use an environment variable
      model: "gemini-pro"
      # Optional Gemini-specific settings
      # gemini_project_id: "your-gcp-project-id"
      # gemini_location: "us-central1"

      # Common LLM parameters (refer to Gemini documentation for exact behavior)
      # temperature: 0.7
      # top_p: 1.0
      # max_tokens: 1024 # This might be 'max_output_tokens' for Gemini

    # You can define other models here (OpenAI, Azure OpenAI, other Gemini instances)
    # another_model:
    #   type: openai_chat
    #   api_key: ${OPENAI_API_KEY}
    #   model: gpt-4o

# Example of referencing this model in a workflow
# (Ensure this key 'my_gemini_chat_model' matches the one defined above)
extract_graph:
  model_id: my_gemini_chat_model
  # ... other settings for extract_graph ...
```

### Gemini Embedding Model (`gemini_embedding`)

For tasks requiring text embeddings (e.g., generating embeddings for text units for vector search), you can configure a Gemini embedding model.

**Key Configuration Parameters:**

*   `type`: Must be set to `gemini_embedding`.
*   `api_key`: Your Google Cloud API key for Gemini. Recommended to use an environment variable.
*   `model`: The specific Gemini embedding model you want to use (e.g., `"embedding-001"`, `"text-embedding-004"`).
*   `gemini_project_id` (Optional): Your Google Cloud Project ID.
*   `gemini_location` (Optional): The Google Cloud location/region.

**Example YAML Configuration:**

```yaml
llm:
  # ... other llm settings ...
  models:
    my_gemini_embedding_model: # You can name your model configuration
      type: gemini_embedding
      api_key: ${GEMINI_API_KEY} # Recommended: use an environment variable
      model: "text-embedding-004" # Or other Gemini embedding model
      # Optional Gemini-specific settings
      # gemini_project_id: "your-gcp-project-id"
      # gemini_location: "us-central1"

    # ... other model definitions ...

# Example of referencing this model for text embedding generation
# (Ensure this key 'my_gemini_embedding_model' matches the one defined above)
text_embedding:
  model_id: my_gemini_embedding_model
  # ... other settings for text_embedding ...
```

## Environment Variables

It is best practice to store your API keys in environment variables rather than directly in your configuration files. GraphRAG supports resolving environment variables using the `${VAR_NAME}` syntax.

Make sure the `GEMINI_API_KEY` (or any other name you choose) environment variable is set in your execution environment.

## Using the Models in Workflows

Once defined in the `models` section, you can reference your Gemini model configurations in various parts of the GraphRAG pipeline by using the key you assigned to it (e.g., `my_gemini_chat_model`, `my_gemini_embedding_model`) in the `model_id` field of the respective workflow step.
```
