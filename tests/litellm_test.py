from litellm import completion, completion_cost, token_counter
import os

# Set your OpenAI API key (make sure it's in your environment)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Define the model
model_name = "deepseek/deepseek-chat"

# Define the prompt messages
messages = [{"role": "user", "content": "Write a short poem about the sea."}]

try:
    # Call the completion API
    response = completion(model=model_name, messages=messages)
    
    # Calculate prompt tokens
    prompt_tokens = token_counter(model=model_name, messages=messages)
    
    # Calculate output tokens
    output_tokens = token_counter(model=model_name, messages=[{"role": "assistant", "content": response.choices[0].message.content}])
    
    # Calculate total tokens
    total_tokens = prompt_tokens + output_tokens
    
    # Calculate cost
    cost = completion_cost(completion_response=response)
    
    # Print results
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Cost (USD): ${cost:.10f}")
    
    # Print the response
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"Error: {e}")
