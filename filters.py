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
def generate_faq_layout(faq_dict: list) -> str:
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


# GRADED CELL

def query_on_faq(query: str, **kwargs) -> dict:
    """
    Constructs a prompt to query an FAQ system and generates a response.

    Parameters:
    - query (str): The query about which the function seeks to provide an answer from the FAQ.
    - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

    Returns:
    - str: The response generated from the LLM based on the input query and FAQ layout.

    """
    ### START CODE HERE ###

    # Make the prompt. Don't forget to add the FAQ_LAYOUT and the query in it!
    prompt = f"""you hav FAQ and its layout is <FAQ>
PROVIDED FAQ: {FAQ_LAYOUT}
</FAQ> and you have to response to query:{query}"""

    # Generate the parameters dict with PROMPT and **kwargs 
    kwargs = generate_params_dict(prompt)

    ### END CODE HERE ###
    
    return kwargs

# GRADED CELL

def query_on_faq(query: str, **kwargs) -> dict:
    """
    Constructs a prompt to query an FAQ system and generates a response.

    Parameters:
    - query (str): The query about which the function seeks to provide an answer from the FAQ.
    - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

    Returns:
    - str: The response generated from the LLM based on the input query and FAQ layout.

    """
    ### START CODE HERE ###

    # Make the prompt. Don't forget to add the FAQ_LAYOUT and the query in it!
    prompt = f"""you hav FAQ and its layout is <FAQ>
PROVIDED FAQ: {FAQ_LAYOUT}
</FAQ> and you have to response to query:{query}"""

    # Generate the parameters dict with PROMPT and **kwargs 
    kwargs = generate_params_dict(prompt)

    ### END CODE HERE ###
    
    return kwargs


# GRADED CELL 

def get_params_for_task(task: str) -> dict:
    """
    Retrieves specific LLM parameters based on the nature of the task.

    This function returns parameter sets optimized for either creative or technical tasks.
    Creative tasks benefit from higher randomness, while technical tasks require more focus and precision.
    A default parameter set is returned for unrecognized task types.

    Parameters:
    - task (str): The nature of the task ('creative' or 'technical').

    Returns:
    - dict: A dictionary containing 'top_p' and 'temperature' settings appropriate for the task.
    """
    ### START CODE HERE ###
    # Define the parameter sets for technical and creative tasks
    PARAMETERS_DICT = {
        "creative": {"top_p": 0.3, 'temperature': 1},
        "technical": {'top_p': 0.7, 'temperature': 0.3}
    }
    
    # Return the corresponding parameter set based on task type
    if task == 'technical':
        param_dict =PARAMETERS_DICT["technical"]
    elif task == 'creative':
        param_dict = PARAMETERS_DICT["creative"]
    else:
        # Fallback to a default parameter set for unrecognized task types
        param_dict =  {'top_p': 0, 'temperature': 0}
    ### END CODE HERE ###
    
    return param_dict
# GRADED CELL

def generate_metadata_from_query(query: str) -> str:
    """
    Generates metadata in JSON format based on a given query to filter clothing items.

    This function constructs a prompt for an LLM to produce a JSON object
    that will guide filtering in a vector database query for clothing items.
    It uses possible values from a predefined set and ensures that only relevant metadata
    is included in the output JSON.

    Parameters:
    - query (str): A description of specific clothing-related needs.

    Returns:
    - str: A JSON string representing metadata with keys such as gender, masterCategory,
      articleType, baseColour, price, usage, and season. Each value in the JSON is a list.
      The price is specified as a dictionary with "min" and "max" keys.
      For unrestricted categories, use ["Any"], and if no price is specified,
      default to {"min": 0, "max": "inf"}.
    """
    ### START CODE HERE ### 

    # Construct the prompt.
    # Include the query, the desired JSON format, and the possible values (pass {values} where needed).
    # Clearly instruct the LLM to include gender, masterCategory, articleType, baseColour, price, usage, and season as keys.
    # Specify that the price key must be a JSON object with "min" and "max" values (0 if no lower bound, "inf" if no upper bound).
    # If no price is set, default to min = None
    prompt = f""" A query will be provided. Based on this query, a vector database will be searched to find relevant clothing items.
    Generate a JSON object containing useful metadata to filter products for this query.
    The possible values for each feature are given in the following JSON: {values}

Provide a JSON containing the features that best match the query (values should be in lists, multiple values possible).
If a price range is mentioned, include a price key specifying the range (between values, greater than, or less than).
Return only the JSON, nothing else. The price key must be a JSON object with "min" and "max" values (use 0 if no lower bound, and "inf" if no upper bound).
Always include the following keys: gender, masterCategory, articleType, baseColour, price, usage, and season.
If no price is specified, set min = 0 and max = inf.
Include only values present in the JSON above.

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

Query: {query}"""

    # Generate the response with generate_with_single_input using PROMPT, temperature=0 (low randomness), and max_tokens=1500
    response = generate_with_single_input(prompt,temperature=0,max_tokens=1500)

    # Extract the content from the response
    content = response['content']

    ### END CODE HERE ###
    
    return content


def parse_json_output(llm_output: str) -> dict:
    """
    Parses a string output from an LLM into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by an LLM.
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the LLM that is expected to be in JSON format.

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


def get_relevant_products_from_query(query: str):
    """
    Retrieve products that are most relevant to a given query by applying filters.

    This function generates filters based on the provided query and uses them to find 
    products that closely match the query criteria. If no filters are applicable or if 
    the initial search returns a small number of products, the function dynamically reduces 
    the filtering constraints based on a predefined order of filter importance.

    Parameters:
    query (str): The query string used to search for relevant products.

    Returns:
    list: A list of product objects that are most relevant to the query. If filters are not effective,
          it adjusts them to ensure a minimum return of products.
    """
    filters = generate_filters_from_query(query)  # Generate filters based on query

    # Check if there are no applicable filters
    if filters is None or len(filters) == 0:
        # Query the collection without filters, using the query text for relevance
        res = products_collection.query.near_text(query, limit=20).objects
        return res

    # Query with filters and limit to top 20 relevant objects
    res = products_collection.query.near_text(query, filters=Filter.all_of(filters), limit=20).objects

    # If the result set is fewer than 10 products, try reducing filters to broaden the search
    importance_order = ['baseColour', 'masterCategory', 'usage', 'masterCategory', 'season', 'gender']

    if len(res) < 10:
        # Iterate through the importance order of filters
        for i in range(len(importance_order)):
            # Create a list of filters that excludes less important ones
            filtered_filters = [x for x in filters if x.target not in importance_order[i+1:]]
            
            # Re-query with the reduced set of filters
            res = products_collection.query.near_text(query, filters=Filter.all_of(filtered_filters), limit=20).objects
            
            # If sufficient products have been found, return early
            if len(res) >= 5:
                return res
        # If there are no enough results, perform a basic near_text with only the query.
        if len(res) < 5:
            res = products_collection.query.near_text(query, limit=20).objects
        
    return res  # Return the final set of relevant products



def generate_items_context(results: list) -> str:
    """
    Compile detailed product information from a list of result objects into a formatted string.

    This function takes a list of results, each containing various product attributes, and constructs 
    a human-readable summary for each product. Each product's details, including ID, name, category, 
    usage, gender, type, and other characteristics, are concatenated into a string that describes 
    all products in the list.

    Parameters:
    results (list): A list of result objects, each having a `properties` attribute that is a dictionary 
                    containing product attributes such as 'product_id', 'productDisplayName', 
                    'masterCategory', 'usage', 'gender', 'articleType', 'subCategory', 
                    'baseColour', 'season', and 'year'.

    Returns:
    str: A multi-line string where each line contains the formatted details of a single product.
         Each product detail includes the product ID, name, category, usage, gender, type, color, 
         season, and year.
    """
    t = ""  # Initialize an empty string to accumulate product information

    for item in results:  # Iterate through each item in the results list
        item = item.properties  # Access the properties dictionary of the current item

        # Append formatted product details to the output string
        t += (
            f"Product ID: {item['product_id']}. "
            f"Product name: {item['productDisplayName']}. "
            f"Product Category: {item['masterCategory']}. "
            f"Product usage: {item['usage']}. "
            f"Product gender: {item['gender']}. "
            f"Product Type: {item['articleType']}. "
            f"Product Category: {item['subCategory']} "
            f"Product Color: {item['baseColour']}. "
            f"Product Season: {item['season']}. "
            f"Product Year: {item['year']}.\n"
        )

    return t  # Return the complete formatted string with product details