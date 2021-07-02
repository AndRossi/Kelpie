import os

ROOT = os.path.realpath(os.path.join(os.path.abspath(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "stored_models")
MAX_PROCESSES = 8