import numpy as np

import pickle
from copy import deepcopy

import time


def softmax(vector): return np.e**vector / (np.e**vector).sum()

class StochasticWorld:
    """
    An episodic environment with stochastic transitions.
    """
    
    def __init__(self,config):
        """
        Constructor method for the StochasticWorld class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            state_space (range): The range of states in this environment's state space
            action_space (range): The range of actions in this environment's action space
            timestep_limit (int): The number of time steps per episode, before the environment terminates

            stochastic_branching_factor (int): The number of possible outcome states that can result from applying an action to a state; if the config value for this parameter is 1, the environment is deterministic
            transition_probabilities_stdev (float): Controls the skewness of the transition probabilities; if the config value for this parameter is 0, all stochastic transitions have uniform probability, whereas the higher the value, the more skewed the probabilities are
            transition_rewards_range (range): The range of transition rewards in this environment's reward space; this range will be sampled from uniformly, when constructing the environment's transition dynamics

            transition_mapping ( dict{ int : dict{ int : dict{} } } ): For every state, maps every action to each possible resulting state, their transition rewards and corresponding transition probabilities, if we were to apply that action to that state

            state_index (int): The state index the environment is currently at
            timestep (int): The current timestep the environment is currently at

            feature_representation (numpy.ndarray): A vector representing the environment's features; a one-hot encoding of the state, concatenated with the current timestep, normalized by the timestep limit

            state_history (list[numpy.ndarray]): Contains a list of past states visited during this episode
            action_history (list[int]): Contains a list of past actions applied to the env (actions are ints in the form of action indices)
            reward_history (list[float]): Contains a list of past transition rewards received from the env

            value_history (list[float]): Contains a list of predicted values for each corresponding past state, outputted by MCTS
            policy_history (list[numpy.ndarray]): Contains a list of action distributions for each corresponding past state, outputted by MCTS
        """
        
        self.state_space = range( config['env']['num_states'] )
        self.action_space = range( config['env']['num_actions'] )
        self.timestep_limit = int( config['env']['timestep_limit'] )

        self.stochastic_branching_factor = config['env']['stochastic_branching_factor']
        self.transition_probabilities_stdev = config['env']['transition_probabilities_stdev']
        self.transition_rewards_range = config['env']['transition_rewards_range']
        
        self.transition_mapping = { state_index : { action_index : {} for action_index in self.action_space } \
                                    for state_index in self.state_space }

        # ensure every state has equal occurrence (although not necessarily equal transition probabilities)
        resulting_states = [state_index for state_index in self.state_space] * len(self.action_space) * self.stochastic_branching_factor
        np.random.shuffle(resulting_states)
        for state_index in self.state_space:
            for action_index in self.action_space:
                self.transition_mapping[state_index][action_index]['resulting_states'] = [ resulting_states.pop() for _ in range(self.stochastic_branching_factor) ]
                # calculate the transition probabilities by softmaxing a sampled vector from a normal distribution
                self.transition_mapping[state_index][action_index]['transition_probabilities'] = softmax( np.random.normal( loc=0 , scale=self.transition_probabilities_stdev,
                                                                                                                            size=self.stochastic_branching_factor) )
                self.transition_mapping[state_index][action_index]['transition_rewards'] = np.random.uniform( *self.transition_rewards_range , self.stochastic_branching_factor )

        self.state_index = None
        self.timestep = None

        self.feature_representation = None

        ###############################################################
        
        self.reset()
        self.state_history = [ self.get_features() ]
        self.action_history = []
        self.reward_history = []

        self.value_history = []
        self.policy_history = []
    def reset(self):
        """
        Reset StochasticWorld environment. Start at a random initial state, set the timestep to 0, and update the feature_representation attribute accordingly.
        
        Args: None

        Returns: None
        """
        
        self.state_index = np.random.choice(self.state_space)
        self.timestep = 0

        self.feature_representation = np.zeros( len(self.state_space) + 1 ) # one-hot state_index vector + timestep feature
        self.feature_representation[self.state_index] = 1
    def apply_action(self,action_index):
        """
        Apply the action_index to the StochasticWorld environment.
        Update the feature_representation attribute and add the transition results to the history attributes, accordingly.
        
        Args:
            action_index (int): Represents an action in the environment's action space

        Returns: None
        """
        
        if self.timestep >= self.timestep_limit: return

        self.feature_representation[self.state_index] = 0 # erase current state_index

        # get resulting state and transition reward
        index = np.random.choice( self.stochastic_branching_factor , p = self.transition_mapping[self.state_index][action_index]['transition_probabilities'] )
        transition_reward = self.transition_mapping[self.state_index][action_index]['transition_rewards'][index]
        self.state_index = self.transition_mapping[self.state_index][action_index]['resulting_states'][index] # update resulting state_index
        self.timestep += 1

        # update feature_representation attribute
        self.feature_representation[self.state_index] = 1 # update resulting one-hot state_index vector
        self.feature_representation[-1] += 1 / self.timestep_limit # update timestep feature

        # add transition results to history attributes
        self.state_history.append( self.get_features() )
        self.action_history.append( action_index )
        self.reward_history.append(transition_reward)
    def sample_random_action(self):
        """
        Randomly sample an action index from the environment's action space.
        
        Args: None

        Returns:
            action_index (int): Represents an action in the environment's action space
        """
        
        return np.random.choice(self.action_space)
    def is_game_over(self):
        """
        Check if the timestep limit is reached. If so, then the environment episode is terminated.

        Args: None

        Returns:
            game_over (bool): True if the environment episode is terminated, False otherwise
        """
        
        return self.timestep >= self.timestep_limit

    def get_features(self):
        """
        Get the feature representation of the environment's current state.
        The features are a one-hot encoded vector denoting the current state, concatenated with the current timestep value, normalized by the timestep limit.
        
        Args: None

        Returns:
            feature_vector (numpy.ndarray): The feature representation of the environment's current state
        """
        
        return self.feature_representation.copy().reshape(1,-1)

    def copy(self):
        """
        Create a deep copy of this StochasticWorld instance.
        This is used to create a new instance of the environment for every episode of self-play, since we store each individual game as a StochasticWorld instance in the replay buffer.

        Args: None

        Returns:
            environment_copy (StochasticWorld): A deepcopy of this StochasticWorld instance
        """
        
        return deepcopy(self)
    def save(self):
        """
        Save this StochasticWorld instance.
        This can be used to test different configurations of learning algorithms on StochasticWorld environments with the same transition dynamics, for comparison.

        Args: None

        Returns: None
        """
        
        with open(f"env_configs/{str(time.time()).replace('.','_')}.pkl",'wb') as file: pickle.dump(self,file)
        
    def __str__(self):
        """
        Return a human-readable string representation of the environment's current state.
        
        Args: None

        Returns:
            string_representation (str): The string representation of the environment's current state
        """
        
        return f'State: {self.state_index}\tTimestep: {self.timestep}\tTimestep limit: {self.timestep_limit}'
    def __repr__(self): return str(self)

def load_env_config(filename):
    """
    Load a saved StochasticWorld instance.
    To be used to test different configurations of learning algorithms on StochasticWorld environments with the same transition dynamics, for comparison.

    Args:
        filename (str): The filename of the StochasticWorld instance you want to load

    Returns:
        environment_instance (StochasticWorld): The loaded StochasticWorld instance
    """
    
    with open(f"env_configs/{filename}.pkl",'rb') as file: return pickle.load(file)
