import os

THRESHOLD = os.getenv("THRESHOLD",0.7) # threshold for voice recognition
TIMEOUT = os.getenv("TIMEOUT",10)  # seconds
EMBEDDING_FILE = "embeddings.json"

RECOGNIZE = os.getenv("RECOGNIZE",False)