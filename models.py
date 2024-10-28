# models.py

import tensorflow as tf
from tensorflow.keras import layers

# Global parameters
num_states = 7      # Number of state features per user
num_actions = 4     # Number of possible actions
num_ues = 5         # Number of User Equipments
num_hidden1 = 64
num_hidden2 = 128
num_hidden3 = 256

def build_model():
    # Input layer
    inputs = layers.Input(shape=(num_ues * num_states,), name='input_layer')
    
    # Common hidden layers
    common1 = layers.Dense(num_hidden1, activation="relu", name='common_dense_1')(inputs)
    common2 = layers.Dense(num_hidden2, activation="relu", name='common_dense_2')(common1)
    common3 = layers.Dense(num_hidden3, activation="relu", name='common_dense_3')(common2)
    
    # Actor network
    logits = layers.Dense(num_ues * num_actions, name='logits')(common3)
    reshaped_logits = layers.Reshape((num_ues, num_actions), name='reshaped_logits')(logits)
    action_probs = layers.Softmax(axis=-1, name='action_probs')(reshaped_logits)
    
    # Critic network
    critic_output = layers.Dense(1, name='critic_output')(common3)
    
    # Define the model with inputs and outputs
    model = tf.keras.Model(inputs=inputs, outputs=[action_probs, critic_output], name='FRL_A3C_Model')
    
    # Compile the model with appropriate loss functions
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'action_probs': 'categorical_crossentropy',
            'critic_output': 'mse'
        },
        metrics={'action_probs': 'accuracy', 'critic_output': 'mae'}
    )
    
    return model 
'''

# models.py

import tensorflow as tf
from tensorflow.keras import layers, models

def create_actor_critic_model(state_shape, action_space):
    """
    Creates an Actor-Critic model.

    Args:
        state_shape (tuple): Shape of the input state.
        action_space (int): Number of possible actions.

    Returns:
        tf.keras.Model: Compiled Actor-Critic model.
    """
    inputs = layers.Input(shape=state_shape)
    
    # Common layers
    common = layers.Dense(128, activation='relu')(inputs)
    common = layers.Dense(128, activation='relu')(common)
    
    # Actor specific layers
    action = layers.Dense(action_space, activation='softmax')(common)
    
    # Critic specific layers
    critic = layers.Dense(1)(common)
    
    model = models.Model(inputs=inputs, outputs=[action, critic])
    return model
'''