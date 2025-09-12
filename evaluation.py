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