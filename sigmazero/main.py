import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import binary_crossentropy,mean_squared_error

import numpy as np

import pickle
import time

from classes import *
from stochastic_world import StochasticWorld,load_env_config


def self_play(network_model,config):
    """
    This is the main function to call.
    Iteratively perform self-play via Monte Carlo Tree Search (MCTS), and then train the network_model.
    Every config['self_play']['test_interval'], test the network_model using greedy action selection and record the average reward received.

    Args:
        network_model (NetworkModel): The network model to be trained
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """

    if config['env']['env_filename']: game_master = load_env_config(f"{config['env']['env_filename']}")
    else: # create a new StocasticWorld instance with randomly generated transition dynamics according to the config parameters
        game_master = StochasticWorld(config)
        game_master.save() # save this instance

    test_rewards = []
    
    # test initial network_model
    test_rewards.append( test(game_master,network_model,config) )
    
    optimizer = Adam(learning_rate=config['train']['learning_rate'],beta_1=config['train']['beta_1'],beta_2=config['train']['beta_2'])

    replay_buffer = ReplayBuffer(config)
    start_time = time.time()
    for num_iter in range( 1 , int(config['self_play']['num_games'])+1 ):
        game = game_master.copy()
        game.reset()

        # self-play
        while not game.is_game_over():
            action_index = mcts(game,network_model,get_temperature(num_iter),config)
            game.apply_action(action_index)

        # training
        replay_buffer.add(game)
        train(network_model,replay_buffer,optimizer,config)

        # save progress
        if (num_iter % config['self_play']['test_interval']) == 0:
            print(f'Iteration: {num_iter}\tTime elapsed: {(time.time()-start_time)/60} minutes')

            # test current network_model
            test_rewards.append( test(game_master,network_model,config) )
            with open(f"test_rewards/test_rewards_{config['model']['stochastic_branching_factor']}.pkl",'wb') as file: pickle.dump(test_rewards,file)

def mcts(game,network_model,temperature,config): 
    """
    Perform Monte Carlo Tree Search (MCTS) on the current environment state, and return an action index that indicates what action to take.

    Args:
        game (StochasticWorld): The StochasticWorld object, containing the current state of the environment
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        temperature (float): Controls the level of exploration of MCTS (the lower the number, the greedier the action selection)
        config (dict): A dictionary specifying parameter configurations

    Returns:
        action_index (int): Represents an action in the environment's action space
    """
    
    root_node = Node(0)
    root_node.expand_root_node(game.get_features(),network_model)
    
    min_q_value,max_q_value = root_node.value,root_node.value # keep track of min and max mean-Q values to normalize them during selection phase
    # this is for environments that have unbounded Q-values, otherwise the prior could potentially have very little influence over selection, if Q-values are large
    for _ in range(int(config['mcts']['num_simulations'])):
        current_node = root_node

        # SELECT a leaf node
        search_path = [root_node] # node0, ... (includes the final leaf node)
        action_history = [] # action0, ...
        while current_node.is_expanded:
            # total_num_visits need to be at least 1
            # otherwise when selecting for child nodes that haven't been visited, their priors won't be taken into account, because it'll be multiplied by total_num_visits in the UCB score, which is zero
            total_num_visits = max( 1 , sum([ current_node.children[i].num_visits for i in range(len(current_node.children)) ]) )

            action_index = np.argmax([ current_node.children[i].get_ucb_score(total_num_visits,min_q_value,max_q_value,config) for i in range(len(current_node.children)) ])
            current_node = current_node.children[action_index]

            search_path.append(current_node)
            action_history.append(action_index) # append action_index to action history, which will be converted to an action matrix before feeding it into the network_model's dynamics function

        # EXPAND selected leaf node
        parent_action_matrix = np.zeros( ( search_path[-2].hidden_state.shape[0] , config['env']['num_actions'] ) ) # copy the action vector for each node in the parent node set
        parent_action_matrix[ : , action_history[-1] ] = 1 # convert the action_index to an action matrix, to be fed into the network_model's dynamics function
        current_node.expand_node( search_path[-2] , parent_action_matrix , network_model )

        # BACKPROPAGATE the bootstrapped value (approximated by the network_model.prediction_function) to all nodes in the search_path
        value = current_node.value
        for node in reversed(search_path):
            node.cumulative_value += value
            node.num_visits += 1

            node_q_value = node.cumulative_value / node.num_visits
            min_q_value , max_q_value = min(min_q_value,node_q_value) , max(max_q_value,node_q_value) # update min and max values

            value = node.transition_reward + config['self_play']['discount_factor'] * value # updated for parent node in next iteration of the loop

    # SAMPLE an action proportional to the visit count of the child nodes of the root node
    total_num_visits = sum([ root_node.children[i].num_visits for i in range(len(root_node.children)) ])
    policy = np.array( [ root_node.children[i].num_visits/total_num_visits for i in range(len(root_node.children)) ] )

    if temperature == None: # take the greedy action (to be used during test time)
        action_index = np.argmax(policy)
    else: # otherwise sample (to be used during training)
        policy = (policy**(1/temperature)) / (policy**(1/temperature)).sum()
        action_index = np.random.choice( range(network_model.action_size) , p=policy )

    # update StochasticWorld environment search statistics
    game.value_history.append( root_node.cumulative_value/root_node.num_visits ) # use the root node's MCTS value as the ground truth value when training
    game.policy_history.append(policy) # use the MCTS policy as the ground truth value when training

    return action_index

def train(network_model,replay_buffer,optimizer,config):
    """
    Train the network_model by sampling episode trajectories from the replay_buffer.

    Args:
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        replay_buffer (ReplayBuffer): Contains a history of the most recent episodes of self-play
        optimizer (tensorflow.python.keras.optimizer_v2.adam.Adam): The optimizer used to update the network_model weights
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    # for every game in sample batch, unroll and update network_model weights for config['train']['num_unroll_steps'] time steps
    with tf.GradientTape() as tape:

        loss = 0
        for game in replay_buffer.sample():

            game_length = len(game.reward_history)
            sampled_index = np.random.choice( range(game_length) ) # sample an index position from the length of reward_history

            hidden_state = network_model.representation_function(game.state_history[sampled_index])
            # first we get the hidden state representation using the representation function
            # then we iteratively feed the hidden state into the dynamics function with the corresponding action, as well as feed the hidden state into the prediction function
            # we then match these predicted values to the true values
            # note we don't call the prediction function on the initial hidden state representation given by the representation function, since there's no associating predicted transition reward to match the true transition reward
            # this is because we don't / shouldn't have access to the previous action that lead to the initial state

            if (sampled_index+config['train']['num_unroll_steps']) < game_length: num_unroll_steps = int(config['train']['num_unroll_steps'])
            else: num_unroll_steps = game_length-1-sampled_index

            current_transition_probabilities = tf.ones( (1,1) ) # the default transition probability for the root state of this sampled game trajectory is 1
            for start_index in range( sampled_index , sampled_index+num_unroll_steps ):
                # can only be unrolled up to the second-last time step, since every time step (start_index), we are predicting and matching values that are one time step into the future (start_index+1)

                ### get predictions ###
                parent_action_matrix = np.zeros( ( hidden_state.shape[0] , config['env']['num_actions'] ) ) # copy the action vector for each node, represented by the hidden_state
                parent_action_matrix[ : , game.action_history[start_index] ] = 1 # convert the action_index to an action matrix, to be fed into the network_model's dynamics function
                hidden_states,transition_rewards,transition_probabilities = network_model.dynamics_function( [ hidden_state , parent_action_matrix ] )

                hidden_state = tf.reshape( hidden_states , ( -1 , config['model']['hidden_state_size'] ) ) # convert ( sample x ( branching_factor x hidden_state_dimension ) ) to ( (sample x branching_factor) x hidden_state_dimension )
                current_transition_probabilities = tf.reshape( tf.multiply( current_transition_probabilities , transition_probabilities ) , (-1,1) ) # multiply (parent_node,1) along the axis=1 of (parent_node x child_nodes), then flatten to (all_child_nodes,1)
                pred_transition_reward = tf.reduce_sum( tf.multiply( tf.reshape( transition_rewards , (-1,1) ) , current_transition_probabilities ) ) # reshape (parent_node,child_node) to (all_child_nodes,1) and multiply by corresponding transition probabilities, then sum them all up
                
                policies,values = network_model.prediction_function( hidden_state )
                pred_policy = tf.reduce_sum( tf.multiply( policies , current_transition_probabilities ) , axis=0 ) # multiply (all_child_nodes,action_index) action probabilities with (all_child_nodes,1) transition probabilities along the axis=1, then sum up all expected action probabilities for each action_index
                pred_value = tf.reduce_sum( tf.multiply( values , current_transition_probabilities ) ) # multiply (all_child_nodes,1) values with (all_child_nodes,1) transition probabilities, then sum them all up
                # the new hidden_state outputted by the dynamics function is at time step (start_index+1)
                # pred_reward is the transition reward outputted by the dynamics function by taking action_(start_index) at state_(start_index)
                # pred_policy and pred_value are the predicted values of state_(start_index+1), using the prediction function
                # therefore, pred_reward, pred_policy and pred_value are all at time step (start_index+1)

                ### make targets ###
                if (game_length - start_index - 1) >= config['train']['num_bootstrap_timesteps']: # bootstrap using transition rewards and mcts value for final bootstrapped time step
                    true_value = sum([ game.reward_history[i] * ( config['self_play']['discount_factor']**(i-start_index) ) for i in range( start_index, int( start_index+config['train']['num_bootstrap_timesteps'] ) ) ]) + \
                                 game.value_history[ start_index + int(config['train']['num_bootstrap_timesteps']) ] * ( config['self_play']['discount_factor']**(config['train']['num_bootstrap_timesteps']) )
                    # using game.reward_history[start_index] actually refers to reward_(start_index+1), since game.reward_history is shifted by 1 time step forward
                    # if the last reward we use is at game.reward_history[end_index], then the value we use to bootstrap is game.value_history[end_index+1]
                    # but since game.reward_history is shifted, we end up actually using reward_(end_index+1) and value_(end_index+1)
                    # this means we get the transition reward going into state_(end_index+1) and the bootstrapped value at state_(end_index+1)
                    # therefore the variable true_value is at time step (start_index+1)
                else: # don't bootstrap; use only transition rewards until termination
                    true_value = sum([ game.reward_history[i] * ( config['self_play']['discount_factor']**(i-start_index) ) for i in range(start_index,game_length) ])

                true_transition_reward = game.reward_history[start_index] # since game.reward_history is shifted, this transition reward is actually at time step (start_index+1)
                true_policy = game.policy_history[start_index+1] # we need to match the pred_policy at time step (start_index+1) so we need to actually index game.policy_history at (start_index+1)

                ### calculate loss ###
                ##### ***NOTE*** BINARY CROSS ENTROPY IS OKAY TO USE???? INSTEAD OF REGULAR CROSS ENTROPY (the first time of BCE is the same CE, what about the second term???)
                    ### technically both terms result are equal in the sense that they are both a*log(a), which means the cross entropy for both terms should be at the lowest value anyways?

                # reshape the predicted values so that they're in a matrix
                loss += (1/num_unroll_steps) * \
                        ( mean_squared_error( true_transition_reward , tf.reshape( pred_transition_reward , (1,1) ) ) + \
                          mean_squared_error( true_value , tf.reshape( pred_value , (1,1) ) ) + \
                          binary_crossentropy( true_policy , tf.reshape( pred_policy , (1,-1) ) ) ) # take the average loss among all unroll steps
        loss += tf.reduce_sum(network_model.representation_function.losses) + tf.reduce_sum(network_model.dynamics_function.losses) + tf.reduce_sum(network_model.prediction_function.losses) # regularization loss

    ### update network_model weights ###
    grads = tape.gradient( loss, [ network_model.representation_function.trainable_variables, network_model.dynamics_function.trainable_variables, network_model.prediction_function.trainable_variables ] )
    optimizer.apply_gradients( zip( grads[0], network_model.representation_function.trainable_variables ) )
    optimizer.apply_gradients( zip( grads[1], network_model.dynamics_function.trainable_variables ) )
    optimizer.apply_gradients( zip( grads[2], network_model.prediction_function.trainable_variables ) )

def get_temperature(num_iter):
    """
    This function regulates exploration vs exploitation when selecting actions during self-play.
    Given the current interation number of the learning algorithm, return the temperature value to be used by MCTS. 

    Args:
        num_iter (int): The number of iterations that have passed for the learning algorithm

    Returns:
        temperature (float): Controls the level of exploration of MCTS (the lower the number, the greedier the action selection)
    """
    
    # as num_iter increases, temperature decreases, and actions become greedier
    if num_iter < 100: return 3
    elif num_iter < 200: return 2
    elif num_iter < 300: return 1
    elif num_iter < 400: return .5
    elif num_iter < 500: return .25
    elif num_iter < 600: return .125
    else: return .0625

def test(game_master,network_model,config):
    """
    Using a trained network_model, greedily play config['test']['num_test_games'] games, and return a list of the game histories.
    If config['test']['record'] is True, record the game renderings.

    Args:
        game_master (StochasticWorld): The StochasticWorld instance with the transition dynamics you want to test; each test episode will use an identical deep copy of this instance
        network_model (NetworkModel): The network model will be used for inference to conduct MCTS
        config (dict): A dictionary specifying parameter configurations

    Returns:
        average_reward (float): The average reward the agent received during the test games
    """

    print('=========== TESTING ===========')

    total_rewards = 0
    start_time = time.time()
    for _ in range( int(config['test']['num_test_games']) ):
        game = game_master.copy()
        game.reset()

        while not game.is_game_over():
            action_index = mcts(game,network_model,None,config) # set temperature value to None, so MCTS always returns the greedy action
            game.apply_action(action_index)

        total_rewards += sum(game.reward_history)

    average_reward = total_rewards / config['test']['num_test_games']
    print(f"Average reward: {average_reward}\tTesting time: {time.time()-start_time} seconds\n")

    return average_reward

if __name__ == '__main__':
    config = { 'env': { 'env_filename': '1642458975_8878424',
                        'num_states': 100, # the number of states in the environment's state space
                        'num_actions': 3, # the number of actions in the environment's action space
                        'timestep_limit': 50, # the number of timesteps in an episode before the environment terminates
                        'stochastic_branching_factor': 2, # the number of possible states that can result from applying an action to a state
                        'transition_probabilities_stdev': 1e-2, # the skewness of the transition probabilities; if 0, all stochastic transitions have uniform probability, whereas the higher the value, the more skewed the probabilities are
                        'transition_rewards_range': (-1,1) }, # the range of transition rewards in this environment's reward space
               'model': { 'representation_function': { 'num_layers': 2, # number of hidden layers
                                                       'num_neurons': 32, # number of hidden units per layer
                                                       'activation_function': 'relu', # activation function for every hidden layer
                                                       'regularizer': L2(1e-3) }, # regularizer for every layer
                          'dynamics_function': { 'num_layers': 2,
                                                 'num_neurons': 32,
                                                 'activation_function': 'relu',
                                                 'regularizer': L2(1e-3) },
                          'prediction_function': { 'num_layers': 2,
                                                   'num_neurons': 32,
                                                   'activation_function': 'relu',
                                                   'regularizer': L2(1e-3) },
                          'hidden_state_size': 32, # size of hidden state representation
                          'stochastic_branching_factor': 2 }, # the branching factor of the model, which determines the output size of the dynamics function; this value affects how many stochastic branches are created during MCTS
               'mcts': { 'num_simulations': 10, # number of simulations to conduct, every time we call MCTS
                         'c1': 1.25, # for regulating MCTS search exploration (higher value = more emphasis on prior value and visit count)
                         'c2': 19625 }, # for regulating MCTS search exploration (higher value = lower emphasis on prior value and visit count)
               'self_play': { 'num_games': 1000, # number of games the agent plays to train on
                              'discount_factor': 1.0, # used when backpropagating values up mcts, and when calculating bootstrapped value during training
                              'test_interval': 25 }, # how often to test the current learned network_model weights, using greedy action selection
               'replay_buffer': { 'buffer_size': 1e2, # size of the buffer
                                  'sample_size': 1e1 }, # how many episode trajectories we sample from the buffer when training the agent
               'train': { 'num_bootstrap_timesteps': 500, # number of timesteps in the future to bootstrap true value
                          'num_unroll_steps': 1e1, # number of timesteps to unroll to match action trajectories for each episode sample
                          'learning_rate': 1e-3, # learning rate for Adam optimizer
                          'beta_1': 0.9, # parameter for Adam optimizer
                          'beta_2': 0.999 }, # parameter for Adam optimizer
               'test': { 'num_test_games': 1000 }, # number of times to test the agent using greedy actions
               'seed': 0
               }

    tf.random.set_seed(config['seed'])
    np.random.seed(config['seed'])

    with tf.device('/CPU:0'):
        network_model = NetworkModel(config)
        self_play(network_model,config)
