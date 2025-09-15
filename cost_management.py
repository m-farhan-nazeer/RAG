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