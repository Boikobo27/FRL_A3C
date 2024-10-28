import tensorflow as tf
from models import build_model
import pickle
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def test_serialization():
    try:
        model = build_model()
        weights = model.get_weights()
        serialized_weights = pickle.dumps(weights)
        encoded_weights = base64.b64encode(serialized_weights).decode('utf-8')
        decoded_weights = base64.b64decode(encoded_weights)
        loaded_weights = pickle.loads(decoded_weights)
        model.set_weights(loaded_weights)
        logging.info("Serialization and deserialization test passed.")
    except Exception as e:
        logging.error(f"Serialization test failed: {e}")

if __name__ == "__main__":
    test_serialization()
