"""
Module 1 – Environment Setup (Auto-add missing keys)
Creates .env if missing, adds any missing required keys, and validates.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

REQUIRED_KEYS = {
    "OPENAI_API_KEY": "your_openai_api_key_here",
    "LANGSMITH_API_KEY": "your_langsmith_api_key_here",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": "rag_course",
    "ANTHROPIC_API_KEY": "your_anthropic_api_key_here",
    "GOOGLE_API_KEY": "your_google_api_key_here",
}


def create_env_if_missing():
    """Create .env with placeholders if it does not exist."""
    if not ENV_PATH.exists():
        with open(ENV_PATH, "w") as f:
            for key, value in REQUIRED_KEYS.items():
                f.write(f"{key}={value}\n")
        print(f"Created {ENV_PATH} with placeholder keys.")


def add_missing_keys():
    """Append any required keys that are missing from .env."""
    with open(ENV_PATH, "r") as f:
        lines = f.readlines()

    existing_keys = {
        line.split("=", 1)[0].strip()
        for line in lines
        if "=" in line and not line.startswith("#")
    }

    missing = [key for key in REQUIRED_KEYS if key not in existing_keys]
    if missing:
        with open(ENV_PATH, "a") as f:
            for key in missing:
                f.write(f"{key}={REQUIRED_KEYS[key]}\n")
                print(f"Added missing key: {key}")
    else:
        print("All required keys are present in .env.")


def validate_env():
    """Check that all required keys are set (non-empty) after loading .env."""
    load_dotenv(ENV_PATH)
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        print("❌ Missing values for:", ", ".join(missing))
        print("Please edit .env and replace placeholders with real keys.")
        return False
    print("✅ All required environment variables are set.")
    return True


def main():
    create_env_if_missing()
    add_missing_keys()
    if validate_env():
        print("Environment setup complete.")
    else:
        print("Fix the missing values and rerun.")


if __name__ == "__main__":
    main()