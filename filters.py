import json
from weaviate.classes.query import Filter
import weaviate
import joblib


import unittests
import flask_app
import weaviate_server
from utils import (
    ChatWidget,
    generate_with_single_input,
    generate_params_dict
)

client = weaviate.connect_to_local(port=8079, grpc_port=50050)

# GRADED CELL 

def check_if_faq_or_product(query: str) -> str:
    """
    Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

    Parameters:
    - query (str): The instruction or query to be labeled as either FAQ or product-related.

    Returns:
    - str: The label 'FAQ' if the prompt is classified as a frequently asked question, 'Product' if it relates to product information, or
      None if the label is inconclusive.
    """
    ### START CODE HERE ###

    # Set the hardcoded prompt. Remember to include the query, clear instructions (explicitly tell the LLM to return FAQ or Product)
    # Include examples of question / desired label pairs.

    prompt = f"""
       You have to  determine whether a given instruction prompt is related to a FAQ or a Product.
       Product-related answers are specific to product information or require using product details to answer. Products are clothes from a store. 
        An FAQ question addresses common inquiries and provides answers to help users find the information they need.
        Examples:
        Is there a refund for incorrectly bought clothes? Label: FAQ
        Tell me about the cheapest T-shirts that you have. Label: Product
        Do you have blue T-shirts under 100 dollars? Label: Product
        I bought a T-shirt and I didn't like it. How can I get a refund? Label: FAQ

        Return only one of the two labels: FAQ or Product.
        
        Instruction: {query}"""
    
        
        

        
        
           


    # Get the kwargs dictionary to call the LLM, with PROMPT as prompt, low temperature (0.3 - 0.5)
    # The function call is generate_params_dict, pass the PROMPT and the correct temperature
    kwargs = generate_params_dict(prompt, temperature=0.4)

    # Call generate_with_single_input with **kwargs
    response = generate_with_single_input(**kwargs)
    # Get the label by accessing the 'content' key of the response dictionary

    label = response['content']

    ### END CODE HERE ###
    
    return label
