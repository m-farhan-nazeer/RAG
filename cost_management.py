import json
from weaviate.classes.query import Filter
import weaviate
import joblib
import pandas as pd


import flask_app
import weaviate_server
import unittests
import json
from utils import (
    ChatWidget, 
    generate_with_single_input,
    parse_json_output,
    get_filter_by_metadata,
    generate_filters_from_query,
    process_and_print_query,
    print_properties,
    make_url
)


client = weaviate.connect_to_local(port=8079, grpc_port=50050)



import phoenix as px
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode


# Launch the lab and the URL
make_url()
session = px.launch_app()


# Setting up the telemetry

phoenix_project_name = "chatbot"

# With phoenix, we just need to register to get the tracer provider with the appropriate endpoint. 
# Different from the ungraded lab, you will NOT use auto_instrument = True, as there are LLM calls not needed to be traced (examples, calls within unittests etc.)

tracer_provider_phoenix = register(project_name=phoenix_project_name, endpoint="http://127.0.0.1:6006/v1/traces")

# Retrieve a tracer for manual instrumentation
tracer = tracer_provider_phoenix.get_tracer(__name__)


def generate_params_dict(
    prompt: str,
    temperature: float = 1.0,
    role: str = 'user',
    top_p: float = 1.0,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
) -> dict:
    """
    Generates a dictionary of parameters for calling a Language Learning Model (LLM),
    allowing for the customization of several key options that can affect the output from the model. 

    Args:
        prompt (str): The input text that will be provided to the model to guide text generation.
        temperature (float): A value between 0 and 1 that controls the randomness of the model's output; 
            lower values result in more repetitive and deterministic results, while higher values enhance randomness.
        role (str): The role designation to be used in context, typically identifying the initiator of the interaction.
        top_p (float): A value between 0 and 1 that manages diversity through the technique of nucleus sampling; 
            this parameter limits the set of considered words to the smallest possible while maintaining 'top_p' cumulative probability.
        max_tokens (int): The maximum number of tokens that the model is allowed to generate in response, where a token can 
            be as short as one character or as long as one word.
        model (str): The specific model identifier to be utilized for processing the request. This typically specifies both 
            the version and configuration of the LLM to be employed.

    Returns:
        dict: A dictionary containing all specified parameters which can then be used to configure and execute a call to the LLM.
    """
    # Create the dictionary with the necessary parameters
    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model
    }
    return kwargs


# GRADED CELL 
def check_if_faq_or_product(query, simplified = False):
    """
    Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

    Parameters:
    - query (str): The instruction or query that needs to be labeled as either FAQ or Product related.
    - simplified (bool): If True, uses a simplified prompt.

    Returns:
    - str: The label 'FAQ' if the prompt is deemed a frequently asked question, 'Product' if it is related to product information, or
      None if the label is inconclusive.
    """
 
    # If not simplified, uses a more complex prompt
    if not simplified:
        PROMPT = f"""Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
        Product related answers are answers specific about product information or that needs to use the products to give an answer.
        Examples:
                Is there a refund for incorrectly bought clothes? Label: FAQ
                Where are your stores located?: Label: FAQ
                Tell me about the cheapest T-shirts that you have. Label: Product
                Do you have blue T-shirts under 100 dollars? Label: Product
                What are the available sizes for the t-shirts? Label: FAQ
                How can I contact you via phone? Label: FAQ
                How can I find the promotions? Label: FAQ
                Give me ideas for a sunny look. Label: Product
        Return only one of the two labels: FAQ or Product, nothing more.
        Query to classify: {query}
                 """

    ##############################################
    ######### GRADED PART STARTS HERE ############
    ##############################################
    
    ### START CODE HERE ###

    # If simlpified, uses a simplified prompt.
    else:
        PROMPT =PROMPT = f"""Label the following instruction as an FAQ related or a product related answer.
         Product related answers are answers specific about product information
            Examples:
                Is there a refund for incorrectly bought clothes? Label: FAQ
                Tell me about the cheapest T-shirts that you have. Label: Product
                Where are your stores located?: Label: FAQ
                Do you have blue T-shirts under 100 dollars? Label: Product
                How can I find the promotions? Label: FAQ
                Give me ideas for a sunny look. Label: Product
        Return only one FAQ or Product.
        Query to classify: {query}
                 """

        
    ### END CODE HERE ###

    ##############################################
    ######### GRADED PART ENDS HERE ############
    ##############################################
        
    with tracer.start_as_current_span("routing_faq_or_product", openinference_span_kind = 'tool') as span:
        span.set_input(str({"query":query, "simplified": simplified}))
        
        # Get the kwargs dictinary to call the llm, with PROMPT as prompt, low temperature (0 or near 0) and max_tokens = 10
        kwargs = generate_params_dict(PROMPT, temperature = 0, max_tokens = 10)

        # Call generate_with_single_input with **kwargs
        with tracer.start_as_current_span("router_call", openinference_span_kind = 'llm') as router_span:
            router_span.set_input(kwargs)
            try:
                response = generate_with_single_input(**kwargs) 
            except Exception as error:
                router_span.record_exception(error)
                router_span.set_status(Status(StatusCode.ERROR))
            else:
                # OpenInference Semantic Conventions for computing Costs
                router_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                router_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                router_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                router_span.set_attribute("llm.model_name", response['model'])
                router_span.set_attribute("llm.provider", 'together.ai')
                router_span.set_output(response)
                router_span.set_status(Status(StatusCode.OK))
        
    
        # Get the Label by accessing the content key of the response dictionary
        label = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']
        span.set_output(str({"label": label, 'total_tokens':total_tokens}))
        span.set_status(Status(StatusCode.OK))

        # Improvement to prevent cases where LLM outputs more than one word
        if 'faq' in label.lower():
            label = 'FAQ'
        elif 'product' in label.lower():
            label = 'Product'
        else:
            label = 'undefined'
    
        return label, total_tokens
    

queries = [
    'What is your return policy?', 
    'Give me three examples of blue T-shirts you have available.', 
    'How can I contact the user support?', 
    'Do you have blue Dresses?',
    'Create a look suitable for a wedding party happening during dawn.'
]

labels = ['FAQ', 'Product', 'FAQ', 'Product', 'Product']

for query, correct_label in zip(queries, labels):
    # Call check_if_faq_or_product and store the results
    response_std, tokens_std = check_if_faq_or_product(query, simplified=False)
    response_simp, tokens_simp = check_if_faq_or_product(query, simplified=True)
    
    # Print results
    process_and_print_query(query, correct_label, response_std, tokens_std, response_simp, tokens_simp)


@tracer.tool
def generate_faq_layout(faq_dict):
    """
    Generates a formatted string layout for a list of FAQs.

    This function iterates through a dictionary of frequently asked questions (FAQs) and constructs
    a string where each question is followed by its corresponding answer and type.

    Parameters:
    - faq_dict (list): A list of dictionaries, each containing keys 'question', 'answer', and 'type' 
      representing an FAQ entry.

    Returns:
    - str: A string representing the formatted layout of FAQs, with each entry on a separate line.
    """
    # Initialize an empty string
    t = ""
    
    # Iterate over every FAQ question in the FAQ list
    for f in faq_dict:
        # Append the question with formatted string (remember to use f-string and access the values as f['question'], f['answer'] and so on)
        # Also, do not forget to add a new line character (\n) at the end of each line.
        t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n" 

    return t