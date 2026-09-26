import os

# Eval traffic would drown out real traces in Langfuse. This runs before any levi
# module is imported (and before load_dotenv, which never overrides existing vars).
# Set LEVI_TRACE_EVALS=1 to trace eval runs anyway.
if os.getenv("LEVI_TRACE_EVALS") != "1":
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
