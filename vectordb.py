from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from typing import List
from tqdm import tqdm
import joblib
import weaviate
import re
from weaviate.util import generate_uuid5
from pprint import pprint
import os

import flask_app
from utils import (
    suppress_subprocess_output, 
    generate_with_single_input, 
    print_object_properties
)


with suppress_subprocess_output():
    client = weaviate.connect_to_embedded(
        persistence_data_path="./.collections",
        environment_variables={
            "ENABLE_API_BASED_MODULES": "true", # Enable API based modules 
            "ENABLE_MODULES": 'text2vec-transformers, reranker-transformers', # We will be using a transformer model
            "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5000/", # The endpoint the weaviate API will be using to vectorize
            "RERANKER_INFERENCE_API":"http://127.0.0.1:5000/" # The endpoint the weaviate API will be using to rerank
        }
    )