from tensorflow.keras.models import Model,save_model,load_model
from tensorflow.keras.layers import Input,Dense,Concatenate

import numpy as np

import os


class NetworkModel: # neural network model
    """
    A class that contains the representation, dynamics and prediction network.
    These networks are trained during agent self-play.
    """
    
    def __init__(self,config):
        """
        Constructor method for the NetworkModel class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            representation_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in the game state as input, and outputs a hidden state representation
            dynamics_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in a hidden state representation and action vector as input, and outputs the resulting hidden state representation and predicted transition reward
            prediction_function (tensorflow.python.keras.engine.functional.Functional): A neural network that takes in a hidden state representation as input, and outputs a predicted value and policy

            action_size (int): The number of possible actions in the game env's action space
        """
        self.action_size = config['env']['num_actions']
        self.stochastic_branching_factor = config['model']['stochastic_branching_factor']

        # building representation function layers
        obs_input_layer = Input( config['env']['num_states'] + 1 ) # one-hot encode state space and include timestep feature
        hidden_layer = Dense(config['model']['representation_function']['num_neurons'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(obs_input_layer)
        for _ in range(config['model']['representation_function']['num_layers']):
            hidden_layer = Dense(config['model']['representation_function']['num_neurons'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(hidden_layer)
        hidden_state_output_layer = Dense(config['model']['hidden_state_size'],activation=config['model']['representation_function']['activation_function'],bias_initializer='glorot_uniform',
                                          kernel_regularizer=config['model']['representation_function']['regularizer'],bias_regularizer=config['model']['representation_function']['regularizer'])(hidden_layer)
        
        self.representation_function = Model(obs_input_layer,hidden_state_output_layer)

        # building dynamics function layers
        hidden_state_input_layer = Input(config['model']['hidden_state_size'])
        action_input_layer = Input(config['env']['num_actions'])
        concat_layer = Concatenate()([hidden_state_input_layer,action_input_layer])
        hidden_layer = Dense(config['model']['dynamics_function']['num_neurons'],activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(concat_layer)
        for _ in range(config['model']['dynamics_function']['num_layers']):
            hidden_layer = Dense(config['model']['dynamics_function']['num_neurons'],activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        # output stochastic_branching_factor hidden state representations, stochastic_branching_factor corresponding transition rewards and stochastic_branching_factor corresponding transition probabilities
        hidden_state_output_layer = Dense( config['model']['hidden_state_size'] * self.stochastic_branching_factor ,activation=config['model']['dynamics_function']['activation_function'],bias_initializer='glorot_uniform',
                                          kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        transition_reward_output_layer = Dense(self.stochastic_branching_factor,activation='linear',bias_initializer='glorot_uniform',
                                               kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        transition_probability_output_layer = Dense(self.stochastic_branching_factor,activation='softmax',bias_initializer='glorot_uniform',
                                                    kernel_regularizer=config['model']['dynamics_function']['regularizer'],bias_regularizer=config['model']['dynamics_function']['regularizer'])(hidden_layer)
        
        self.dynamics_function = Model([hidden_state_input_layer,action_input_layer],[hidden_state_output_layer,transition_reward_output_layer,transition_probability_output_layer]) # output (hidden_state_representations,transition_rewards,transition_probabilities)

        # building prediction function layers
        hidden_state_input_layer = Input(config['model']['hidden_state_size'])
        hidden_layer = Dense(config['model']['prediction_function']['num_neurons'],activation=config['model']['prediction_function']['activation_function'],bias_initializer='glorot_uniform',
                             kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_state_input_layer)
        for _ in range(config['model']['prediction_function']['num_layers']):
            hidden_layer = Dense(config['model']['prediction_function']['num_neurons'],activation=config['model']['prediction_function']['activation_function'],bias_initializer='glorot_uniform',
                                 kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        policy_output_layer = Dense(config['env']['num_actions'],activation='softmax',bias_initializer='glorot_uniform',
                                    kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        value_output_layer = Dense(1,activation='linear',bias_initializer='glorot_uniform',
                                   kernel_regularizer=config['model']['prediction_function']['regularizer'],bias_regularizer=config['model']['prediction_function']['regularizer'])(hidden_layer)
        
        self.prediction_function = Model(hidden_state_input_layer,[policy_output_layer,value_output_layer])

class Node:
    """
    A class that represents the nodes used in Monte Carlo Tree Search.
    """
    
    def __init__(self,prior):
        """
        Constructor method for the Node class.
        
        Args:
            prior (float): The prior probability assigned to this Node upon instantiation, obtained from the prediction function of the NetworkModel (this value is supposed to represent how promising this Node is, before exploration)
            
        Attributes:
            prior (float): The prior probability assigned to this Node upon instantiation, obtained from the argument

            hidden_state (numpy.ndarray): The hidden state representation of all stochastic states from this Node, obtained from either the representation or dynamics function of the NetworkModel
            transition_reward (float): The predicted expected transition reward as a result of transitioning to this Node from the parent Node, obtained from the dynamics function of the NetworkModel
            transition_probabilities (numpy.ndarray): The predicted transition probabilities of transitioning to every stochastic state in this node from the root
            policy (numpy.ndarray): The predicted action distribution, obtained from applying the prediction function to this Node's hidden_state (the values will serve as priors for the children of this Node)
            value (float): The predicted expected value of this Node, obtained from applying the prediction function to this Node's hidden_state

            is_expanded (bool): True if this Node is expanded, False otherwise
            children (list[Node]): Contains a list of this Node's children

            cumulative_value (float): Every simulation of MCTS that contains this Node in its search path, the leaf node's predicted value is backpropagated up and accumulated in this cumulative_value
            num_visits (int): The number of simulations of MCTS that contained this Node in its search path (this value is used to divide the cumulative_value to get the mean Q-value)
        """
        
        self.prior = prior # prior probability given by the output of the prediction function of the parent node
        
        self.hidden_state = None # from dynamics function; a (all_nodes x hidden_state_features) matrix, where each row is a possible stochastic representation of this node
        self.transition_reward = 0 # from dynamics function
        self.transition_probabilities = None # from dynamics function; a (all_nodes x 1) vector, where each value denotes the probability of reaching that specific stochastic state representation from the root node
        self.policy = None # from prediction function; a vector, where each value denotes the prior probability for each action
        self.value = None # from prediction function

        self.is_expanded = False
        self.children = []

        self.cumulative_value = 0 # divide this by self.num_visits to get the mean Q-value of this node
        self.num_visits = 0
    def expand_node(self,parent_node,parent_action,network_model):
        """
        Expand this Node. Use the dynamics function on parent_hidden_state and parent_action to get this Node's hidden state representations and corresponding transition rewards for all resulting stochastic states.
        Then use the prediction function on this Node's hidden state representations to get this Node's values and the prior probabilities for every stochastic state representation of this Node.
        
        Args:
            parent_node (Node): This node's parent; the parent node's hidden state representation will be passed into the dynamics function, and the parent node's transition probabilities will be used to calculate this node's transition probabilities
            parent_action (numpy.ndarray): A (num_parent_nodes x action_space) matrix, where each row is an identical one-hot vector, denoting the corresponding action index taken from this Node's parent to get to this Node
            network_model (NetworkModel): The NetworkModel's dynamics function and prediction function will be used to expand this Node

        Returns: None
        """

        # get hidden state representation and transition reward
        hidden_states,transition_rewards,transition_probabilities = network_model.dynamics_function( [ parent_node.hidden_state , parent_action ] )
        self.hidden_state = hidden_states.numpy().reshape( -1 , parent_node.hidden_state.shape[1] ) # convert ( sample x ( branching_factor x hidden_state_dimension ) ) to ( (sample x branching_factor) x hidden_state_dimension )
        self.transition_probabilities = ( parent_node.transition_probabilities * transition_probabilities.numpy() ).reshape(-1,1) # multiply (parent_node,1) along the axis=1 of (parent_node x child_nodes), then flatten to (all_child_nodes,1)
        self.transition_reward = ( transition_rewards.numpy().reshape(-1,1) * self.transition_probabilities ).sum() # reshape (parent_node,child_node) to (all_child_nodes,1) and multiply by corresponding transition probabilities, then sum them all up

        # get predicted policy and value
        policies,values = network_model.prediction_function( self.hidden_state )
        self.policy = ( policies.numpy() * self.transition_probabilities ).sum(axis=0) # multiply (all_child_nodes,action_index) action probabilities with (all_child_nodes,1) transition probabilities along the axis=1, then sum up all expected action probabilities for each action_index
        self.value = ( values.numpy() * self.transition_probabilities ).sum() # multiply (all_child_nodes,1) values with (all_child_nodes,1) transition probabilities, then sum them all up

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append( Node(self.policy[action]) )
            
        self.is_expanded = True
    def expand_root_node(self,current_state,network_model):
        """
        Expand this Node. This function should only be called on root nodes, and assumes that this Node is a root node.
        Use the representation function on the game env's current_state to get this Node's hidden state representation.
        Then use the prediction function on this Node's hidden state representation to get this Node's value and the prior probabilities of this Node's children.
        
        Args:
            current_state (numpy.ndarray): The numpy array representation of the game env's current state
            network_model (NetworkModel): The NetworkModel's representation function and prediction function will be used to expand this Node

        Returns: None
        """
        
        # similar to self.expand_node() method, except representation function is used to get this node's hidden state
        # therefore there's no corresponding predicted transition reward for the root node

        # get hidden state representation
        hidden_state = network_model.representation_function(current_state).numpy()
        self.hidden_state = hidden_state
        self.transition_reward = 0 # no transition reward for the root node
        self.transition_probabilities = np.array([[1]]) # the transition probability for the root node is always 1; this value will be used by all child nodes to calculate their corresponding transition probabilities, based on the search path

        # get predicted policy and value
        policy,value = network_model.prediction_function( self.hidden_state )
        self.policy = policy.numpy().reshape(-1)
        self.value = value[0][0].numpy() # convert to scalar

        # instantiate child Node's with prior values, obtained from the predicted policy
        for action in range(network_model.action_size):
            self.children.append( Node(self.policy[action]) )
            
        self.is_expanded = True
    def get_ucb_score(self,visit_sum,min_q_value,max_q_value,config):
        """
        Calculate the modified UCB score of this Node. This value will be used when selecting Nodes during MCTS simulations.
        The UCB score balances between exploiting Nodes with known promising values, and exploring Nodes that haven't been searched much throughout the MCTS simulations.
        
        Args:
            visit_sum (int): The total number of visits across all child Node's of this Node's parent
            min_q_value (float): The minimum Q-value found across all Nodes selected by MCTS across all simulations (used for min-max normalization)
            max_q_value (float): The maximum Q-value found across all Nodes selected by MCTS across all simulations (used for min-max normalization)
            config (dict): A dictionary specifying parameter configurations

        Returns:
            ucb_score (float): This value is calculated across all child Nodes and MCTS selects the child Node with the highest UCB score to add to the search path
        """
        
        normalized_q_value = self.transition_reward + config['self_play']['discount_factor'] * self.cumulative_value / max(self.num_visits,1)
        if min_q_value != max_q_value: normalized_q_value = (normalized_q_value - min_q_value) / (max_q_value - min_q_value) # min-max normalize q-value, to make sure q-value is in the interval [0,1]
        # if min and max value are equal, we would end up dividing by 0

        return normalized_q_value + \
               self.prior * np.sqrt(visit_sum) / (1 + self.num_visits) * \
               ( config['mcts']['c1'] + np.log( (visit_sum + config['mcts']['c2'] + 1) / config['mcts']['c2'] ) )
        
class ReplayBuffer:
    """
    A class that stores the history of the most recent games of self-play.
    """
    
    def __init__(self,config):
        """
        Constructor method for the ReplayBuffer class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations

        Attributes:
            buffer (list[StochasticWorld]): Buffer that stores StochasticWorld objects.
            buffer_size (int): Indicates the maximum size of the buffer
            sample_size (int): Indicates how many game trajectories to sample from the buffer when we call the sample() method
        """
        
        self.buffer = [] # list of StochasticWorld objects, that contain the state, action, reward, MCTS policy, and MCTS value history
        self.buffer_size = int(config['replay_buffer']['buffer_size'])
        self.sample_size = int(config['replay_buffer']['sample_size'])
    def add(self,game):
        """
        Add the game to the ReplayBuffer. Remove the oldest StochasticWorld entry if the size of the buffer exceeds buffer_size (which is set by the config parameter upon instantiation).
        
        Args:
            game (StochasticWorld): The StochasticWorld game instance to add to the ReplayBuffer

        Returns: None
        """
        
        if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
        self.buffer.append(game)
    def sample(self): # sample a number of game trajectories from self.buffer, specified by the config parameter
        """
        Sample a number of game trajectories from the buffer equal to sample_size (which is set by the config parameter upon instantiation).
        
        Args: None

        Returns:
            game_samples (list[StochasticWorld]): A list of sampled StochasticWorld games to be used to train the NetworkModel weights
        """
        
        if len(self.buffer) <= self.sample_size: return self.buffer.copy()
        return np.random.choice(self.buffer,size=self.sample_size,replace=False).tolist()

