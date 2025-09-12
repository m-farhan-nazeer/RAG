import utils
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


# Define a resource with attributes that describe your application
# Here, we're setting the service name to identify what is being traced
resource = Resource(attributes={
    "service.name": "Test Service"
})

# Set up the tracer provider that will manage and provide tracers
# 'TracerProvider' is initialized with the resource we just defined
trace.set_tracer_provider(TracerProvider(resource=resource))

# Create a console exporter to output spans to the console for demonstration purposes
# In a real-world scenario, you might use an OTLP exporter to send spans to a tracing system
console_exporter = ConsoleSpanExporter()

# Set up a span processor to handle the spans
# SimpleSpanProcessor sends each span to the exporter as soon as it is finished
span_processor = SimpleSpanProcessor(console_exporter)

# Add the span processor to the tracer provider to start processing spans immediately
trace.get_tracer_provider().add_span_processor(span_processor)

# Obtain a tracer for the current module to create and manage spans
tracer = trace.get_tracer(__name__)



def retrieve(query, fail=False):
    # Start a span to trace the retrieval process
    with tracer.start_as_current_span("retrieving_documents") as span:
        # Log the event of starting retrieval
        span.add_event("Starting retrieve")
        # Record the input query as an attribute for visibility
        span.set_attribute("input.query", query)
        try:
            # Simulate a retrieval failure if 'fail' is True
            if fail:
                raise ValueError(f"Retrieve failed for query: {query}")

            # Simulated list of retrieved documents
            retrieved_docs = ['retrieved doc1', 'retrieved doc2', 'retrieved doc3']
            # Record details about each retrieved document
            for i, doc in enumerate(retrieved_docs):
                span.set_attribute(f"retrieval.documents.{i}.document.id", i)
                span.set_attribute(f"retrieval.documents.{i}.document.content", doc)
                span.set_attribute(f"retrieval.documents.{i}.document.metadata", f"Metadata for document {i}")
        except Exception as e:
            # If an exception occurs, log and set the span status to indicate an error
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            # Reraise the exception for handling by the caller
            raise

        # Mark the span as successful if no error was raised
        span.set_status(Status(StatusCode.OK))
        return retrieved_docs