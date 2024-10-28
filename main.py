# main.py

from server import Server
from agent import Agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    # Initialize the server
    server = Server()

    # Number of agents
    num_agents = 5
    agents = []

    # Start agents
    for i in range(num_agents):
        agent = Agent(agent_id=i)
        agents.append(agent)
        agent.start()
        logging.info(f"Started Agent {i}")

    # Wait for all agents to finish
    for agent in agents:
        agent.join()
        logging.info(f"Agent {agent.agent_id} has finished.")

    logging.info("Training completed.")

if __name__ == "__main__":
    main()

