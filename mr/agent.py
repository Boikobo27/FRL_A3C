# agent.py

import threading
import numpy as np
import tensorflow as tf
from models import build_model
from mobile_env.scenarios.small import MComSmall
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import base64
import pickle
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Global parameters
num_states = 7      # Number of state features per user
num_actions = 4     # Number of possible actions
num_ues = 5         # Number of User Equipments
gamma = 0.99        # Discount factor
max_steps_per_episode = 50
max_episodes = 1000

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

class Agent(threading.Thread):
    def __init__(self, agent_id):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.local_model = build_model()
        self.env = MComSmall()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.total_step = 1
        self.grads = None
        self.lock = threading.Lock()
        # Initial synchronization with the global model
        self.synchronize_with_server(episode=0, initial=True)
        logging.info(f"Agent {self.agent_id} initialized.")

    def run(self):
        for episode in range(1, max_episodes + 1):
            state = self.env.reset()
            state = np.array(state, dtype='float32')
            done = False
            mem = []
            total_reward = 0
            self.total_step = 1  # Reset total_step for each episode

            while not done and self.total_step < max_steps_per_episode:
                # Flatten state for all users
                state_all_users = state.flatten().reshape(1, -1)  # Shape: (1, num_ues * num_states)

                # Get action probabilities and critic values from local model
                action_probs, values = self.local_model(state_all_users, training=True)
                # action_probs: (1, num_ues, num_actions)
                # values: (1, 1)

                # Compute actions using TensorFlow operations
                action_probs_tf = tf.squeeze(action_probs, axis=0)  # Shape: (num_ues, num_actions)
                values_tf = tf.squeeze(values, axis=-1)  # Shape: (1,)

                actions = []
                for i in range(num_ues):
                    action_prob = action_probs_tf[i]
                    
                    # Enforce normalization to handle numerical precision issues
                    action_prob = action_prob / tf.reduce_sum(action_prob)
                    
                    # Sample action based on probabilities
                    action = np.random.choice(num_actions, p=action_prob.numpy())
                    actions.append(action)

                actions = np.array(actions)

                # Interact with the environment
                next_state, reward, done, _ = self.env.step(actions)
                next_state = np.array(next_state, dtype='float32')
                total_reward += reward

                # Store experience
                mem.append((state_all_users, actions, reward, done))
                state = next_state
                self.total_step += 1

            # Update the local model
            if mem:
                self.train_local_model(mem, episode)
            else:
                logging.warning(f"Agent {self.agent_id}, Episode {episode}, No experiences to train.")

            logging.info(f"Agent {self.agent_id}, Episode {episode}, Total Reward: {total_reward}")

            # Synchronize with the server after each episode
            self.synchronize_with_server(episode=episode, initial=False)

        self.env.close()
        logging.info(f"Agent {self.agent_id} has completed all episodes.")

    def train_local_model(self, mem, episode):
        try:
            with tf.GradientTape() as tape:
                # Unpack experiences
                all_states = []
                all_actions = []
                all_rewards = []
                all_dones = []

                for (states, actions, reward, done) in mem:
                    all_states.extend(states)
                    all_actions.extend(actions)
                    all_rewards.append(reward)
                    all_dones.append(done)

                all_states = np.array(all_states, dtype='float32').reshape(-1, num_ues * num_states)
                all_actions = np.array(all_actions)
                all_rewards = np.array(all_rewards, dtype='float32')
                all_dones = np.array(all_dones, dtype='bool')

                # Compute discounted rewards
                discounted_rewards = self.discount_rewards(all_rewards, all_dones)
                discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)  # Shape: (batch_size,)

                # Get predictions from the local model
                action_probs, values = self.local_model(all_states, training=True)  # Ensure training=True

                # Compute advantages
                values = tf.squeeze(values, axis=-1)  # Shape: (batch_size,)
                advantages = discounted_rewards - values  # Shape: (batch_size,)

                # Compute critic loss (Mean Squared Error)
                critic_loss = tf.reduce_mean(tf.square(advantages))

                # Compute actor loss (Policy Gradient)
                # Reshape actions to (batch_size, num_ues)
                actions = all_actions.reshape(-1, num_ues)  # Shape: (batch_size, num_ues)
                # Create one-hot encoding for actions
                actions_one_hot = tf.one_hot(actions, num_actions)  # Shape: (batch_size, num_ues, num_actions)
                # Compute log probabilities
                log_probs = tf.math.log(action_probs + 1e-10)  # Shape: (batch_size, num_ues, num_actions)
                # Multiply log probabilities by actions_one_hot
                selected_log_probs = tf.reduce_sum(log_probs * actions_one_hot, axis=-1)  # Shape: (batch_size, num_ues)
                # Sum over UEs
                selected_log_probs = tf.reduce_sum(selected_log_probs, axis=-1)  # Shape: (batch_size,)
                # Compute actor loss
                actor_loss = -tf.reduce_mean(selected_log_probs * advantages)  # Scalar

                # Total loss
                total_loss = actor_loss + critic_loss

                # Optional: Add regularization or entropy loss here if needed

            # Calculate gradients and update the local model
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            # Handle None gradients by replacing them with zero tensors
            grads = [g if g is not None else tf.zeros_like(w) for g, w in zip(grads, self.local_model.trainable_weights)]
            self.optimizer.apply_gradients(zip(grads, self.local_model.trainable_weights))

            # Optionally, log gradient norms for debugging
            for i, grad in enumerate(grads):
                grad_norm = tf.norm(grad).numpy()
                logging.debug(f"Agent {self.agent_id}, Layer {i}, Gradient Norm: {grad_norm}")

            # Store gradients for synchronization
            self.grads = grads
            logging.info(f"Agent {self.agent_id}, Episode {episode}, Training complete.")
        except Exception as e:
            logging.error(f"Agent {self.agent_id}, Episode {episode}, Training failed: {e}")

    def discount_rewards(self, rewards, dones):
        discounted = np.zeros_like(rewards, dtype='float32')
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                cumulative = 0.0
            cumulative = cumulative * gamma + rewards[i]
            discounted[i] = cumulative
        return discounted

    def synchronize_with_server(self, episode, initial=False):
        with self.lock:
            try:
                # Load the latest global model
                version, global_weights = load_latest_global_model()
                if global_weights is not None:
                    if initial:
                        # Set local model weights to global weights without updating
                        self.local_model.set_weights(global_weights)
                        logging.info(f"Agent {self.agent_id}, Initial synchronization with global model version {version}.")
                    else:
                        # Apply gradients to the global model
                        global_model = build_model()
                        global_model.set_weights(global_weights)

                        # Apply local gradients to global model
                        self.optimizer.apply_gradients(zip(self.grads, global_model.trainable_weights))

                        # Update the global model in InfluxDB
                        new_weights = global_model.get_weights()
                        new_version = version + 1
                        save_global_model(new_version, new_weights)

                        # Update local model with new global weights
                        self.local_model.set_weights(new_weights)
                        logging.info(f"Agent {self.agent_id}, Episode {episode}, Synchronized with global model version {new_version}")
                else:
                    logging.error(f"Agent {self.agent_id}, Episode {episode}, Failed to load global model")
            except Exception as e:
                logging.error(f"Agent {self.agent_id}, Episode {episode}, Synchronization failed: {e}")
