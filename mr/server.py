# server.py

import threading
import logging
import base64
import pickle
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from models import build_model
import tensorflow as tf
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# InfluxDB configuration
influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
token = os.getenv("INFLUXDB_TOKEN", "WF8WfrvPCr7i1URqmjOWiyta4qzVfoH9ZwWebACOMJW9xfIvKkH_AJhSZAWB902kWviQXd3rxCVhW_dGmnwXLg==")
org = os.getenv("INFLUXDB_ORG", "FRL_a3c")
bucket = os.getenv("INFLUXDB_BUCKET", "a3c")     # Replace with your bucket name

client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

def load_latest_global_model():
    try:
        query = f'''
        from(bucket:"{bucket}")
          |> range(start: -1d)
          |> filter(fn: (r) => r._measurement == "global_model")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n:1)
        '''
        logging.debug(f"Executing query to load global model: {query}")
        result = query_api.query(org=org, query=query)
        logging.debug(f"Query result: {result}")
        if result and len(result) > 0:
            records = result[0].records
            logging.debug(f"Records found: {len(records)}")
            if records and len(records) > 0:
                record = records[0]
                encoded_weights = record.get_value_by_key("weights")
                version = int(record.get_value_by_key("version"))
                serialized_weights = base64.b64decode(encoded_weights)
                weights = pickle.loads(serialized_weights)
                logging.info(f"Loaded global model version {version}.")
                return version, weights
            else:
                logging.warning("No records found in the query result.")
        else:
            logging.warning("No results returned from the query.")
    except Exception as e:
        logging.error(f"Exception in load_latest_global_model: {e}")
    return None, None

def save_global_model(version, weights):
    try:
        serialized_weights = pickle.dumps(weights)
        encoded_weights = base64.b64encode(serialized_weights).decode('utf-8')
        point = Point("global_model") \
            .field("version", version) \
            .field("weights", encoded_weights) \
            .time(version, WritePrecision.S)
        write_api.write(bucket=bucket, org=org, record=point)
        logging.info(f"Global model version {version} saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save global model: {e}")

class Server:
    def __init__(self):
        self.lock = threading.Lock()
        self.initialize_global_model()
        logging.info("Server initialized.")

    def initialize_global_model(self):
        version, weights = load_latest_global_model()
        if weights is None:
            # Initialize and store the global model
            global_model = build_model()
            weights = global_model.get_weights()
            version = 1
            save_global_model(version, weights)
            logging.info("Initialized and stored the global model.")
        else:
            logging.info(f"Global model version {version} retrieved.")

    def aggregate_gradients(self, agent_gradients):
        """
        Aggregates gradients received from agents and updates the global model.
        This function should be called after collecting gradients from all agents.
        """
        with self.lock:
            try:
                version, global_weights = load_latest_global_model()
                if global_weights is not None:
                    global_model = build_model()
                    global_model.set_weights(global_weights)

                    # Average the gradients
                    averaged_gradients = []
                    for grads_list in zip(*agent_gradients):
                        averaged_gradients.append(tf.reduce_mean(grads_list, axis=0))

                    # Apply the averaged gradients
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    optimizer.apply_gradients(zip(averaged_gradients, global_model.trainable_weights))

                    # Update the global model in InfluxDB
                    new_weights = global_model.get_weights()
                    new_version = version + 1
                    save_global_model(new_version, new_weights)
                    logging.info(f"Aggregated gradients and updated global model to version {new_version}.")
                else:
                    logging.error("Failed to load global model for aggregation.")
            except Exception as e:
                logging.error(f"Exception during gradient aggregation: {e}")
