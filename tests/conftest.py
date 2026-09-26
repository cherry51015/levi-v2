import os

# Tests must never send traces to Langfuse. load_dotenv() doesn't override variables
# that already exist, so blanking them here (before levi is imported) wins over .env.
for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ[key] = ""
