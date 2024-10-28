# database.py

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import base64
import pickle
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# Configure InfluxDB connection using environment variables
influxdb_url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
token = os.getenv("INFLUXDB_TOKEN", "WF8WfrvPCr7i1URqmjOWiyta4qzVfoH9ZwWebACOMJW9xfIvKkH_AJhSZAWB902kWviQXd3rxCVhW_dGmnwXLg==")
org = os.getenv("INFLUXDB_ORG", "FRL_a3c")
bucket = os.getenv("INFLUXDB_BUCKET", "a3c")

if not token:
    logging.error("InfluxDB token is not set. Please set INFLUXDB_TOKEN in your environment variables.")
    raise ValueError("InfluxDB token is missing.")

client = InfluxDBClient(url=influxdb_url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_global_model(version, weights):
    """
    Saves the global model to InfluxDB.

    Args:
        version (int): Version number of the global model.
        weights (list): List of model weights.
    """
    try:
        # Serialize and encode weights
        serialized_weights = pickle.dumps(weights)
        encoded_weights = base64.b64encode(serialized_weights).decode('utf-8')

        # Create a point with the current UTC time as the timestamp
        point = Point("global_model") \
            .field("version", version) \
            .field("weights", encoded_weights) \
            .time(datetime.utcnow(), WritePrecision.S)

        # Write the point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)
        logging.info(f"Global model version {version} saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save global model: {e}")
        raise e  # Allow retry

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_latest_global_model():
    """
    Loads the latest global model from InfluxDB.

    Returns:
        tuple: (version, weights) if successful, (None, None) otherwise.
    """
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

                # Log all available fields for debugging
                logging.debug(f"Available fields in record: {record.values}")
                print(f"Available fields in record: {record.values}")  # For manual inspection

                # Access fields using the 'values' attribute
                encoded_weights = record.values.get("weights")
                version = record.values.get("version")

                if encoded_weights is None or version is None:
                    logging.error("Required fields 'weights' or 'version' are missing in the record.")
                    return None, None

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
        raise e  # Allow retry
    return None, None
