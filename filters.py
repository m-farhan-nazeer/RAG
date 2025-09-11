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