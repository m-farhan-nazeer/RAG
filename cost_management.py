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