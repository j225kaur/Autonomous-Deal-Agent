from src.utils.io import load_config
import os

# Create a dummy env_var if it doesn't exist (though we know it does)
if not os.path.exists("env_var"):
    with open("env_var", "w") as f:
        f.write("CHAT_MODEL=api:openai\n")

cfg = load_config()
print(f"CHAT_MODEL env: {os.environ.get('CHAT_MODEL')}")
# Note: load_config doesn't return CHAT_MODEL, it's read from env by adapters.py
# But load_dotenv should have set os.environ
print(f"CHAT_MODEL after load_config: {os.environ.get('CHAT_MODEL')}")
