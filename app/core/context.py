import contextvars

# Stores a unique Request ID for the current execution context
request_id_var = contextvars.ContextVar("request_id", default=None)
