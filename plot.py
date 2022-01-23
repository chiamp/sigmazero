import pickle

import matplotlib.pyplot as plt


def plot(env_stochastic_branching_factor,test_interval):
    with open('muzero/test_rewards/test_rewards.pkl','rb') as file: muzero_rewards = pickle.load(file)
    with open('sigmazero/test_rewards/test_rewards_1.pkl','rb') as file: sigmazero_rewards_1 = pickle.load(file)
    with open('sigmazero/test_rewards/test_rewards_2.pkl','rb') as file: sigmazero_rewards_2 = pickle.load(file)
    with open('sigmazero/test_rewards/test_rewards_3.pkl','rb') as file: sigmazero_rewards_3 = pickle.load(file)

##    x_values = range(0,len(muzero_rewards)*test_interval,test_interval)

    # initial graph
    plt.plot( range( 0 , len(muzero_rewards)*test_interval , test_interval ) , muzero_rewards,
              color='blue' , label='muzero' )
    plt.plot( range( 0 , len(sigmazero_rewards_2)*test_interval , test_interval ) , sigmazero_rewards_2 ,
              color='orange' , label=f'sigmazero (stochastic_branching_factor=2)' )
    
    plt.legend()
    plt.title(f"SigmaZero and MuZero performance on StochasticWorld with a stochastic branching factor of {env_stochastic_branching_factor}")
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')

    plt.show()
    plt.close()

    # initial graph plus other stochastic branching factors
    plt.plot( range( 0 , len(muzero_rewards)*test_interval , test_interval ) , muzero_rewards,
              color='blue' , label='muzero' )
    plt.plot( range( 0 , len(sigmazero_rewards_2)*test_interval , test_interval ) , sigmazero_rewards_2 ,
              color='orange' , label=f'sigmazero (stochastic_branching_factor=2)' )
    
    plt.plot( range( 0 , len(sigmazero_rewards_1)*test_interval , test_interval ) , sigmazero_rewards_1 ,
              color='red' , linestyle='--' , label=f'sigmazero (stochastic_branching_factor=1)' )
    plt.plot( range( 0 , len(sigmazero_rewards_3)*test_interval , test_interval ) , sigmazero_rewards_3 ,
              color='green' , linestyle='--' , label=f'sigmazero (stochastic_branching_factor=3)' )

    plt.legend()
    plt.title(f"SigmaZero and MuZero performance on StochasticWorld with a stochastic branching factor of {env_stochastic_branching_factor}")
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')

    plt.show()
    plt.close()
        
    
if __name__ == '__main__':
    plot(env_stochastic_branching_factor=2,test_interval=25)
