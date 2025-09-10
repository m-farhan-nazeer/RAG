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
