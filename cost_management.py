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



from tqdm import tqdm
from weaviate.util import generate_uuid5
# Set up a batch process with specified fixed size and concurrency
with faq_collection.batch.fixed_size(batch_size=20, concurrent_requests=5) as batch:
    # Iterate over a subset of the dataset
    for document in tqdm(faq):
        # Generate a UUID based on the chunk text for unique identification
        uuid = generate_uuid5(document['question'])

        # Add the chunk object to the batch with properties and UUID
        batch.add_object(
            properties=document,
            uuid=uuid,
        )



# GRADED CELL 

def query_on_faq(query, simplified = False, **kwargs):
    """
    Constructs a prompt to query an FAQ system and generates a response.

    This function integrates an FAQ layout into the prompt to help generate a suitable answer to the given query
    using a language model. It supports additional keyword arguments to customize the prompt generation process.

    Parameters:
    - query (str): The query about which the function seeks to provide an answer from the FAQ.
    - simplified (bool): If True, uses semantic search to extract a relevant subset of FAQ questions
    - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

    Returns:
    - str: The response generated from the language model based on the input query and FAQ layout.

    """

    
    # If not simplified, generate the faq layout with the entire FAQ questions
    if not simplified:
        # Set the tracer as a chain type, since in non-simplified version, the full FAQ is used
        with tracer.start_as_current_span("query_on_faq", openinference_span_kind="tool") as span:
            
            span.set_input({"query": query, "simplified": simplified})
            faq_layout = generate_faq_layout(faq)
            
            # Generate the prompt
            PROMPT = f"""You will be provided with an FAQ for a clothing store. 
        Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ. 
        <FAQ_ITEMS>
        PROVIDED FAQ: {faq_layout}
        </FAQ_ITEMS>
        Question: {query}
            """ 
            span.set_attribute("prompt", PROMPT)

            # Generate the parameters dict with PROMPT and **kwargs 
            kwargs = generate_params_dict(PROMPT, **kwargs) 

            span.set_attribute("output", str(kwargs))
            span.set_status(Status(StatusCode.OK))
    
            return kwargs
        
   
    
    else:
        with tracer.start_as_current_span("query_on_faq", openinference_span_kind="tool") as span:
            span.set_input({"query": query, "simplified": simplified})
            with tracer.start_as_current_span("retrieve_faq_questions", openinference_span_kind="retriever") as retrieve_span:
                
                ##############################################
                ######### GRADED PART STARTS HERE ############
                ##############################################
                
                ### START CODE HERE ###
                
                # Get the 5 most relevant FAQ objects, in this case limit = None
                results = faq_collection.query.near_text(query=query,limit=5)

                ### END CODE HERE ###

                ##############################################
                ######### GRADED PART ENDS HERE ##############
                ##############################################
                
                # Set the retrieved documents as attributes on the span
                for i, document in enumerate(results.objects): 
                    retrieve_span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid)) 
                    retrieve_span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata)) 
                    retrieve_span.set_attribute( 
                        f"retrieval.documents.{i}.document.content", str(document.properties) 
                    )  
            # Transform the results in a list of dictionary
                results = [x.properties for x in results.objects] 
                # Reverse the order to add the most relevant objects in the bottom, so it gets closer to the end of the input
                results.reverse() 
                # Generate the faq layout with the new list of FAQ questions `results`
                faq_layout = generate_faq_layout(results) 

            # Different prompt to deal with this new scenario. 
            PROMPT = (f"You will be provided with a query for a clothing store regarding FAQ. It will be provided relevant FAQ from the clothing store." 
        f"Answer the query based on the relevant FAQ provided. They are ordered in decreasing relevance, so the first is the most relevant FAQ and the last is the least relevant."  
        f"Answer the instruction based on them. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.\n"  
        f"<FAQ>\n"  
        f"RELEVANT FAQ ITEMS:\n{faq_layout}\n"  
        f"</FAQ>\n" 
        f"Query: {query}")

    
        
            span.set_attribute("prompt", PROMPT)
        
            # Generate the parameters dict with PROMPT and **kwargs 
            kwargs = generate_params_dict(PROMPT, **kwargs) 
        
            span.set_attribute("output", str(kwargs))
            span.set_status(Status(StatusCode.OK))
    
            return kwargs
        


# GRADED CELL 
def decide_task_nature(query, simplified = True):
    """
    Determines the nature of a query, labeling it as either creative or technical.

    This function constructs a prompt for a language model to decide if a given query requires a creative response,
    such as making suggestions or composing ideas, or a technical response, like providing product details or prices.

    Parameters:
    - query (str): The query to be evaluated for its nature.
    - simplified (bool): If True, uses a simplified prompt.

    Returns:
    - str: The label 'creative' if the query requires creative input, or 'technical' if it requires technical information.
    """


    
    if not simplified:
        PROMPT = f"""Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices etc.). Label it as creative or technical.
          Examples:
          Give me suggestions on a nice look for a nightclub. Label: creative
          What are the blue dresses you have available? Label: technical
          Give me three Tshirts for summer. Label: technical
          Give me a look for attending a wedding party. Label: creative
          Give me suggestions on clothes that match a green Tshirt. Label: creative
          I would like a suggestion on which products match a green Tshirt I already have. Label: creative

          Query to be analyzed: {query}. Only output one token with the label
          """

    # If simplified, uses a simplified query

    ##############################################
    ######### GRADED PART STARTS HERE ############
    ##############################################

    ### START CODE HERE ###

    else:
        PROMPT =  PROMPT = f"""Decide if query that requires creativity (creating, composing, making new things) or technical (information about products, prices etc.). Label it as creative or technical.
          Examples:
          Give me suggestions on a nice look for a nightclub. Label: creative
          What are the blue dresses you have available? Label: technical
          Give me three Tshirts for summer. Label: technical
          Give me a look for attending a wedding party. Label: creative


          Query to be analyzed: {query}. Only output one token with the label
          """
    ### END CODE HERE ###

    ##############################################
    ######### GRADED PART ENDS HERE ##############
    ##############################################

    
    with tracer.start_as_current_span("decide_task_nature", openinference_span_kind="tool") as span:
    # Generate the kwards dictionary by passing the PROMPT, low temperature and max_tokens = 1
        span.set_input({"query":query, "simplified": simplified})
        kwargs = generate_params_dict(PROMPT, temperature = 0, max_tokens = 1)

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
    
        return label, total_tokens
    


@tracer.tool
def get_params_for_task(task):
    """
    Retrieves specific language model parameters based on the task nature.

    This function provides parameter sets tailored for creative or technical tasks to optimize
    language model behavior. For creative tasks, higher randomness is encouraged, while technical
    tasks are handled with more focus and precision. A default parameter set is provided for unexpected cases.

    Parameters:
    - task (str): The nature of the task ('creative' or 'technical').

    Returns:
    - dict: A dictionary containing 'top_p' and 'temperature' settings for the specified task.
    """
    # Create the parameters dict for technical and creative tasks
    PARAMETERS_DICT = {"creative": {'top_p': 0.9, 'temperature': 1},
                       "technical": {'top_p': 0.7, 'temperature': 0.3}} 
    
    # If task is technical, return the value for the key technical in PARAMETERS_DICT
    if task == 'technical':
        param_dict = PARAMETERS_DICT['technical'] 

    # If task is creative, return the value for the key creative in PARAMETERS_DICT
    if task == 'creative':
        param_dict = PARAMETERS_DICT['creative'] 

    # If task is a different value, fallback to another set of parameters
    else: # Fallback to a standard value
        param_dict = {'top_p': 0.5, 'temperature': 1} 

    
    return param_dict


def generate_metadata_from_query(query):
    """
    Generates metadata in JSON format based on a given query to filter clothing items.

    This function constructs a prompt for a language model to create a JSON object that will
    guide the filtering of a vector database query for clothing items. It takes possible values from
    a predefined set and ensures only relevant metadata is included in the output JSON.

    Parameters:
    - query (str): The query describing specific clothing-related needs.

    Returns:
    - str: A JSON string representing metadata with keys like gender, masterCategory, articleType,
      baseColour, price, usage, and season. Each value in the JSON is within a list, with prices specified
      as a dict containing "min" and "max" values. Unrestricted keys should use ["Any"] and unspecified
      prices should default to {"min": 0, "max": "inf"}.
    """

    # Set the prompt. Remember to include the query, the desired JSON format, the possible values (passing {values} at some point) 
    # and explain to the LLM what is going on. 
    # Explicitly tell the llm to include gender, masterCategory, ArticleType, baseColour, price, usage and season as keys.
    # Also mention to the llm that price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound)
    # If there is no price set, add min = 0 and max = inf.
    PROMPT = f"""
    One query will be provided. For the given query, there will be a call on vector database to query relevant clothing items. 
    Generate a JSON with useful metadata to filter the products in the query. Possible values for each feature is in the following json: {values}

    Provide a JSON with the features that best fit in the query (can be more than one, write in a list). Also, if present, add a price key, saying if there is a price range (between values, greater than or smaller than some value).
    Only return the JSON, nothing more. price key must be a JSON with "min" and "max" values (0 if no lower bound and inf if no upper bound). 
    Always include gender, masterCategory, articleType, baseColour, price, usage and season as keys. All values must be within lists.
    If there is no price set, add min = 0 and max = inf.
    Only include values that are given in the json above. 
    
    Example of expected JSON:

    {{
    "gender": ["Women"],
    "masterCategory": ["Apparel"],
    "articleType": ["Dresses"],
    "baseColour": ["Blue"],
    "price": {{"min": 0, "max": "inf"}},
    "usage": ["Formal"],
    "season": ["All seasons"]
    }}

    Query: {query}
             """
    with tracer.start_as_current_span("generate_metadata_from_query", openinference_span_kind="tool") as span:
        span.set_input(query)
        with tracer.start_as_current_span("llm_call", openinference_span_kind="llm") as metadata_span:
            # Generate the response with the generate_with_single_input, PROMPT, temperature = 0 (low randomness) and max_tokens = 1500.
            kwargs = {"prompt": PROMPT, 'temperature': 0, "max_tokens": 1500}  # @REPLACE EQUALS None
            metadata_span.set_input(kwargs)
            try:
                response = generate_with_single_input(**kwargs) 
            except Exception as error:
                metadata_span.record_exception(error)
                metadata_span.set_status(Status(StatusCode.ERROR))
            else:
                # OpenInference Semantic Conventions for computing Costs
                metadata_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                metadata_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                metadata_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                metadata_span.set_attribute("llm.model_name", response['model'])
                metadata_span.set_attribute("llm.provider", 'together.ai')
                metadata_span.set_output(response)
                metadata_span.set_status(Status(StatusCode.OK))

        # Get the Label by accessing the content key of the response dictionary
        content = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']
        span.set_output({"content": content, 'total_tokens':total_tokens})
        span.set_status(Status(StatusCode.OK))   

    
    return content, total_tokens


@tracer.tool
def parse_json_output(llm_output):
    """
    Parses a string output from a language model into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by a language model (LLM).
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the language model that is expected to be in JSON format.

    Returns:
    - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

    Exception Handling:
    - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
    """
    try:
        # Since the input might be improperly formatted, ensure any single quotes are removed
        llm_output = llm_output.replace("\n", '').replace("'",'').replace("}}", "}").replace("{{", "{")  # Remove any erroneous structures
        
        # Attempt to parse JSON directly provided it is a properly-structured JSON string
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None

@tracer.tool
def get_filter_by_metadata(json_output: dict | None = None):
    """
    Generate a list of Weaviate filters based on a provided metadata dictionary.

    Parameters:
    - json_output (dict) or None: Dictionary containing metadata keys and their values.

    Returns:
    - list[Filter] or None: A list of Weaviate filters, or None if input is None.
    """
    # If the input dictionary is None, return None immediately
    if json_output is None:
        return None

    # Define a tuple of valid keys that are allowed for filtering
    valid_keys = (
        'gender',
        'masterCategory',
        'articleType',
        'baseColour',
        'price',
        'usage',
        'season',
    )

    # Initialize an empty list to store the filters
    filters = []

    # Iterate over each key-value pair in the input dictionary
    for key, value in json_output.items():
        # Skip the key if it is not in the list of valid keys
        if key not in valid_keys:
            continue

        # Special handling for the 'price' key
        if key == 'price':
            # Ensure the value associated with 'price' is a dictionary
            if not isinstance(value, dict):
                continue

            # Extract the minimum and maximum prices from the dictionary
            min_price = value.get('min')
            max_price = value.get('max')

            # Skip if either min_price or max_price is not provided
            if min_price is None or max_price is None:
                continue

            # Skip if min_price is non-positive or max_price is infinity
            if min_price <= 0 or max_price == 'inf':
                continue

            # Add filters for price greater than min_price and less than max_price
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        else:
            # For other valid keys, add a filter that checks for any of the provided values
            filters.append(Filter.by_property(key).contains_any(value))

    return filters


@tracer.tool
def generate_filters_from_query(query):
    json_string, total_tokens = generate_metadata_from_query(query)
    json_output = parse_json_output(json_string)
    filters = get_filter_by_metadata(json_output)
    return filters, total_tokens