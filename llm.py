import json
import random

from utils import (
    generate_with_single_input, 
    generate_with_multiple_input
)


def generate_params_dict(
    prompt: str, 
    temperature: float = None, 
    role = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
):
    """
    Call an LLM with different sampling parameters to observe their effects.
    
    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use
        
    Returns:
        The LLM response
    """
    
    # Create the dictionary with the necessary parameters
    kwargs = {"prompt": prompt, 'role':role, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, 'model': model} 


    return kwargs

def call_llm_with_context(prompt: str, context: list,  role: str = 'user', **kwargs):
    """
    Calls a language model with the given prompt and context to generate a response.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).

    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """

    # Append the dictionary {'role': role, 'content': prompt} into the context list
    context.append({'role': role, 'content': prompt})

    # Call the llm with multiple input passing the context list and the **kwargs
    response = generate_with_multiple_input(context, **kwargs)

    # Append the LLM response in the context dict
    context.append(response) 
    
    return response


# Example usage
context = [{"role": 'system', 'content': 'You are an ironic but helpful assistant.'}, 
           {'role': 'assistant', 'content': "How can I help you, majesty?"}]
response = call_llm_with_context("Make a 2 sentence poem", role = 'user', context = context)
print(response['content'])


# Now we can keep the conversation
response = call_llm_with_context("Now add two more sentences.", context = context)
print(response['content'])



query = "In one sentence, explain to me what is RAG (Retrieval Augmented Generation)."
# Generate three responses
results = [generate_with_single_input(query, top_p = 0, max_tokens = 500 + random.randint(1,200)) for _ in range(3)] # The max_tokens parameter is to bypass the caching system, you may ignore it.
for i,result in enumerate(results):
    print(f"Call number {i+1}:\nResponse: {result['content']}")



# Generate three responses
results = [generate_with_single_input(query, top_p = 0.8, max_tokens = 500 + random.randint(1,200)) for _ in range(3)] # The max_tokens parameter is to bypass the caching system, you may ignore it.
for i,result in enumerate(results):
    print(f"Call number {i+1}:\nResponse: {result['content']}")


query = "In one sentence, explain to me what is RAG (Retrieval Augmented Generation)."
# Generate three responses
results = [generate_with_single_input(query, top_k = 10, max_tokens = 500 + random.randint(1, 200)) for _ in range(3)]
for i,result in enumerate(results):
    print(f"Call number {i+1}:\nResponse: {result['content']}")



# Generate three responses
results = [generate_with_single_input(query, temperature = t) for t in [0.3, 1.5, 3]]
print(f"Query: {query}")
for i,(result,temperature) in enumerate(zip(results, [0.3,1.5,3])):
    print(f"\033[1mCall number {i+1}.\033[0m \033[1mTemperature = {temperature}\033[0m\nResponse: {result['content']}\n\n\n")



# Generate three responses
query = "List healthy breakfast options."

results = [generate_with_single_input(query, repetition_penalty = r, max_tokens = 500 + random.randint(1,200)) for r in [None, 1.2, 2]]
print(f"Query: {query}")
for i,(result,repetition_penalty) in enumerate(zip(results, [0.3,1.5,3])):
    print(f"\033[1mCall number {i+1}.\033[0m \033[1mRepetition Penalty = {repetition_penalty}\033[0m\nResponse: {result['content']}\n\n\n")



def print_response(response):
    """
    Prints a formatted chatbot response with color-coded roles.

    The function uses ANSI escape codes to apply text styles. Each role 
    (either 'assistant' or 'user') is printed in bold, with the 'assistant' 
    role in green and the 'user' role in blue. The content of the response 
    follows the role name.

    Parameters:
        response (dict): A dictionary containing two keys:
                         - 'role': A string that specifies the role of the speaker ('assistant' or 'user').
                         - 'content': A string with the message content to be printed.
    """
    # ANSI escape codes
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    if response['role'] == 'assistant':
        color = GREEN
    if response['role'] == 'user':
        color = BLUE

    s = f"{BOLD}{color}{response['role'].capitalize()}{RESET}: {response['content']}"
    print(s)