import numpy as np

import pickle
from copy import deepcopy

import time


def softmax(vector): return np.e**vector / (np.e**vector).sum()

class StochasticWorld:
    def __init__(self,config):
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
        self.state_index = np.random.choice(self.state_space)
##        self.timestep = 1
        self.timestep = 0

        self.feature_representation = np.zeros( len(self.state_space) + 1 ) # one-hot state_index vector + timestep feature
        self.feature_representation[self.state_index] = 1
##        self.feature_representation[-1] = 1 / self.timestep_limit
    def apply_action(self,action_index):
        if self.timestep >= self.timestep_limit: return

        self.feature_representation[self.state_index] = 0 # erase current state_index
        
        index = np.random.choice( self.stochastic_branching_factor , p = self.transition_mapping[self.state_index][action_index]['transition_probabilities'] )
        transition_reward = self.transition_mapping[self.state_index][action_index]['transition_rewards'][index]
        self.state_index = self.transition_mapping[self.state_index][action_index]['resulting_states'][index] # update resulting state_index
        self.timestep += 1

        self.feature_representation[self.state_index] = 1 # update resulting one-hot state_index vector
        self.feature_representation[-1] += 1 / self.timestep_limit # update timestep feature
        
##        return transition_reward

        self.state_history.append( self.get_features() )
##        self.action_history.append( np.array([1 if i==action_index else 0 for i in self.action_space]).reshape(1,-1) )
        self.action_history.append( action_index )
        self.reward_history.append(transition_reward)
    def sample_random_action(self): return np.random.choice(self.action_space)
    def is_game_over(self): return self.timestep >= self.timestep_limit

    def get_features(self): return self.feature_representation.copy().reshape(1,-1)

    def copy(self): return deepcopy(self)
    def save(self):
        with open(f"env_configs/{str(time.time()).replace('.','_')}.pkl",'wb') as file: pickle.dump(self,file)
        
    def __str__(self): return f'State: {self.state_index}\tTimestep: {self.timestep}\tTimestep limit: {self.timestep_limit}'
    def __repr__(self): return str(self)

def load_env_config(filename): # the main function is called from the root directory, so we have to reference the sigmazero child directory
    with open(f"env_configs/{filename}.pkl",'rb') as file: return pickle.load(file)
    

if __name__ == '__main__':
    config = { 'env': { 'num_states': 1000,
                        'num_actions': 3,
                        'timestep_limit': 50,
                        'stochastic_branching_factor': 2,
                        'transition_probabilities_stdev': 1e-0,
                        'transition_rewards_range': (-1,1) },
               'seed': 1
               }
    np.random.seed(config['seed'])
    
    s = StochasticWorld(config)
##    from pprint import pprint
##    pprint(s.transition_mapping)

##    s.save()
    
##    s2 = load_env_config('1642298359_2423809')
##    pprint(s2.transition_mapping)
    
##    ds = s.copy()

##    while True:
##        s.reset()
##        while not s.is_game_over():
##            print(s)
##            print(s.get_features())
##            r = s.apply_action( int( input('Action: ') ) )
##            print(f'Reward: {r}\n')
##        print('Game over')
##        print(s)
##        break

##    sm = load_env_config('1642304495_541649')
##    n = 1e5
##    interval = 1e4
##    cr = 0
##    for sim_num in range(1,int(n)+1):
##        s = sm.copy()
##        s.reset()
##        while not s.is_game_over():
##            s.apply_action( s.sample_random_action() )
##        cr += sum(s.reward_history)
##        if sim_num % interval == 0: print(cr/sim_num)
                        
