# SigmaZero
This is a repo where I generalize DeepMind's MuZero reinforcement learning algorithm on stochastic environments, and create an algorithm called SigmaZero (stochastic MuZero). For more details on the algorithm specifics, check out the original [paper](https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D) and my [project](https://github.com/chiamp/muzero-cartpole) on applying the MuZero algorithm on the cartpole environment.

## Table of Contents
* [What is MuZero?](#what-is-muzero)
* [Thoughts](#thoughts)
* [What is gym?](#what-is-gym)
* [MuZero Technical Details](#muzero-technical-details)
* [File Descriptions](#file-descriptions)
* [Additional Resources](#additional-resources)

## Can MuZero work in stochastic environments?
* feature representation equal to state space size

## MuZero

### Functions
MuZero contains 3 functions approximated by neural networks, to be learned from the environment:
* A representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_t) \rightarrow s^0">, which given an observation <img src="https://render.githubusercontent.com/render/math?math=o_t"> from the environment at time step <img src="https://render.githubusercontent.com/render/math?math=t">, outputs the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> of the observation at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=0"> (this hidden state will be used as the root node in MCTS, so its hypothetical time step is zero)
	* The representation function is used in tandem with the dynamics function to represent the environment's state in whatever way the algorithm finds useful in order to make accurate predictions for the reward, value and policy
* A dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^k,a^{k%2B1}) \rightarrow s^{k%2B1},r^{k%2B1}">, which given a hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^k"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k"> and action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k%2B1">, outputs the predicted resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}"> and transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{k%2B1}"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k%2B1">
	* The dynamics function is the learned transition model, which allows MuZero to utilize MCTS and plan hypothetical future actions on future board states
* A prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^k) \rightarrow p^k,v^k">, which given a hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^k">, outputs the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^k"> and value <img src="https://render.githubusercontent.com/render/math?math=v^k"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k">
	* The prediction function is used to limit the search breadth by using the policy output to prioritize MCTS to search for more promising actions, and limit the search depth by using the value output as a substitute for a Monte Carlo rollout

### Algorithm Overview
The MuZero algorithm can be summarized as follows:
* loop for a number of episodes:
	* at every time step <img src="https://render.githubusercontent.com/render/math?math=t"> of the episode:
		* perform Monte Carlo tree search (MCTS)
			* pass the current observation of the environment <img src="https://render.githubusercontent.com/render/math?math=o_t"> to the representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_t) \rightarrow s^0">, and get the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> from the output
			* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^0) \rightarrow p^0,v^0">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^0"> and value <img src="https://render.githubusercontent.com/render/math?math=v^0"> from the output
			* for a number of simulations:
				* select a leaf node based on maximizing UCB score
				* expand the node by passing the hidden state representation of its parent node <img src="https://render.githubusercontent.com/render/math?math=s^k"> and its corresponding action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> into the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^k,a^{k%2B1}) \rightarrow s^{k%2B1},r^{k%2B1}">, and get the predicted resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}"> and transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{k%2B1}"> from the output
				* pass the resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{k%2B1}) \rightarrow p^{k%2B1},v^{k%2B1}">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^{k%2B+1}"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{k%2B1}"> from the output
				* backpropagate the predicted value <img src="https://render.githubusercontent.com/render/math?math=v^{k%2B1}"> up the search path
		* sample an action based on the visit count of each child node of the root node
		* apply the sampled action to the environment and observe the resulting transition reward
	* once the episode is over, save the game trajectory (including the MCTS results) into the replay buffer
	* sample a number of game trajectories from the replay buffer:
		* pass the first observation of the environment <img src="https://render.githubusercontent.com/render/math?math=o_0"> from the game trajectory to the representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_0) \rightarrow s^0"> and get the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> from the output
		* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^0) \rightarrow p^0,v^0">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^0"> and value <img src="https://render.githubusercontent.com/render/math?math=v^0"> from the output
		* for every time step <img src="https://render.githubusercontent.com/render/math?math=t"> in the game trajectory:
			* pass the current hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^t"> and the corresponding action <img src="https://render.githubusercontent.com/render/math?math=a^{t%2B1}"> into the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^t,a^{t%2B1}) \rightarrow s^{t%2B1},r^{t%2B1}">, and get the predicted resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{t%2B1}"> and transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{t%2B1}"> from the output
				* this predicted transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{t%2B1}"> is matched to the actual transition reward target received from the environment
			* pass the resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{t%2B1}"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{t%2B1}) \rightarrow p^{t%2B1},v^{t%2B1}">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^{t%2B1}"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{t%2B1}"> from the output
				* this predicted policy distribution <img src="https://render.githubusercontent.com/render/math?math=p^{t%2B1}"> is matched to the child node visit count distribution outputted by MCTS at that time step in that game trajectory
				* this predicted value <img src="https://render.githubusercontent.com/render/math?math=v^{t%2B1}"> is matched to the value outputted by MCTS at that time step in that game trajectory
			* update the weights of the representation, dynamics and prediction function based on these three targets

## Monte Carlo Tree Search in Stochastic Environments

MCTS requires a model of the environment when expanding leaf nodes during its search. The environment model takes in a state and action and outputs the resulting state and transition reward; this is the functional definition of the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^k,a^{k%2B1}) \rightarrow s^{k%2B1},r^{k%2B1}">, which approximates the true environment model. This works for deterministic environments where there is a single outcome for any action applied to any state.

In stochastic environments, the functional definition of the environment model changes. Given a state and action, the environment model instead outputs a **set** of possible resulting states, transition rewards and the corresponding probabilities of those outcomes occurring. To approximate this environment model, we can re-define the dynamics function as: <img src="https://render.githubusercontent.com/render/math?math=g(s^k,a^{k%2B1}) \rightarrow [s^{k%2B1}_1,...,s^{k%2B1}_b],[r^{k%2B1}_1,...,r^{k%2B1}_b],[\pi^{k%2B1}_1,...,\pi^{k%2B1}_b]">, where <img src="https://render.githubusercontent.com/render/math?math=\pi^{k%2B1}_i"> is the predicted probability that applying action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> to state <img src="https://render.githubusercontent.com/render/math?math=s^k"> results in the predicted state <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}_i"> with transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{k%2B1}_i">.

Given a current state <img src="https://render.githubusercontent.com/render/math?math=s"> and action <img src="https://render.githubusercontent.com/render/math?math=a">, a perfect environment model would output a corresponding probability for every possible transition sequence <img src="https://render.githubusercontent.com/render/math?math=s,a \rightarrow s^',r">, where <img src="https://render.githubusercontent.com/render/math?math=s^'"> is the resulting state and <img src="https://render.githubusercontent.com/render/math?math=r"> is the resulting transition reward. To approximate this with the dynamics function, we would need to define the function to output a number of predicted transitions <img src="https://render.githubusercontent.com/render/math?math=(s^{k%2B1}_i,r^{k%2B1}_i,\pi^{k%2B1}_i)"> equal to all possible transitions of the environment. This requires additional knowledge of the environment's state space, reward space and transition dynamics.

Instead we define a **stochastic branching factor** hyperparameter  <img src="https://render.githubusercontent.com/render/math?math=b"> which sets and limits the number of predicted transitions the dynamics function can output. MCTS can then use this modified dynamics function to expand nodes and account for stochastic outcomes.

Below is an animation illustrating how MCTS is executed in stochastic environments (you can find the individual pictures in `assets/sigmazero_graph_images/` for closer inspection). In this environment, the action space is size 3 and the stochastic branching factor hyperparameter is set to 2.

![Alt text](assets/sigmazero_mcts.gif)

Instead of a single node representing a state, we have a **node set** representing a number of possible states, that are a result of applying a sequence of actions from the state at the root node. For example, <img src="https://render.githubusercontent.com/render/math?math=[s^2_1,s^2_2,s^2_3,s^2_4]"> are the predicted possible states as a result of applying the action <img src="https://render.githubusercontent.com/render/math?math=a_3"> and then <img src="https://render.githubusercontent.com/render/math?math=a_1"> to the state at the root node <img src="https://render.githubusercontent.com/render/math?math=s^0_1">. And thus, <img src="https://render.githubusercontent.com/render/math?math=[r^2_1,r^2_2,r^2_3,r^2_4]"> are the predicted transition rewards, <img src="https://render.githubusercontent.com/render/math?math=[v^2_1,v^2_2,v^2_3,v^2_4]"> are the predicted values and <img src="https://render.githubusercontent.com/render/math?math=[p^2_1,p^2_2,p^2_3,p^2_4]"> are the predicted policy distributions over actions for each corresponding state.

Just like nodes used for MCTS in deterministic environments, node sets also contain a predicted reward, value and policy distribution attribute. Whereas in the former case where the predicted reward would be obtained directly from the dynamics function output and the predicted value and policy distribution would be obtained directly from the prediction function output, the predicted reward, value and policy distribution for node sets are calculated **via expectation**.

More formally, to obtain the predicted reward <img src="https://render.githubusercontent.com/render/math?math=r_N">, value <img src="https://render.githubusercontent.com/render/math?math=v_N"> and policy distribution <img src="https://render.githubusercontent.com/render/math?math=p_N"> for a node set <img src="https://render.githubusercontent.com/render/math?math=N">, we calculate the following:

<img src="https://render.githubusercontent.com/render/math?math=r_N = \sum_{i \in N} \pi_i r_i">

<img src="https://render.githubusercontent.com/render/math?math=v_N = \sum_{i \in N} \pi_i v_i">

<img src="https://render.githubusercontent.com/render/math?math=p_N = [ \sum_{i \in N} \pi_i p_{i1} , ... , \sum_{i \in N} \pi_i p_{ia} ]">

where <img src="https://render.githubusercontent.com/render/math?math=\pi_i"> is the probability of transitioning to the hidden state of node <img src="https://render.githubusercontent.com/render/math?math=i"> in node set <img src="https://render.githubusercontent.com/render/math?math=N"> after applying the corresponding actions to the state of the root node, <img src="https://render.githubusercontent.com/render/math?math=r_i"> is the corresponding transition reward of node <img src="https://render.githubusercontent.com/render/math?math=i"> (obtained from the dynamics function output), <img src="https://render.githubusercontent.com/render/math?math=v_i"> is the corresponding value of node <img src="https://render.githubusercontent.com/render/math?math=i"> (obtained from the prediction function output), and <img src="https://render.githubusercontent.com/render/math?math=p_{ia}"> is the corresponding prior probability of taking action <img src="https://render.githubusercontent.com/render/math?math=a"> at node <img src="https://render.githubusercontent.com/render/math?math=i"> (also obtained from the prediction function output).

Once these three values are computed, MCTS can perform all 4 stages of selection, expansion, simulation and backpropagation on a node set just like it does on a regular node.

## SigmaZero

### Algorithm Overview
The SigmaZero algorithm can be summarized as follows:
* loop for a number of episodes:
	* at every time step <img src="https://render.githubusercontent.com/render/math?math=t"> of the episode:
		* perform Monte Carlo tree search (MCTS)
			* pass the current observation of the environment <img src="https://render.githubusercontent.com/render/math?math=o_t"> to the representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_t) \rightarrow s^0">, and get the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> from the output
			* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^0) \rightarrow p^0,v^0">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^0"> and value <img src="https://render.githubusercontent.com/render/math?math=v^0"> from the output
			* for a number of simulations:
				* select a leaf node set based on maximizing UCB score
				* expand the node set by passing the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^k_i"> for each node <img src="https://render.githubusercontent.com/render/math?math=i"> in the parent node set <img src="https://render.githubusercontent.com/render/math?math=N^k"> and the corresponding action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> into the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^k_i,a^{k%2B1}) \rightarrow [s^{k%2B1}_{i1},...,s^{k%2B1}_{ib}],[r^{k%2B1}_{i1},...,r^{k%2B1}_{ib}],[\pi^{k%2B1}_{i1},...,\pi^{k%2B1}_{ib}]">, and get the predicted resulting hidden state representations <img src="https://render.githubusercontent.com/render/math?math=[s^{k%2B1}_{i1},...,s^{k%2B1}_{ib}]">, transition rewards <img src="https://render.githubusercontent.com/render/math?math=[r^{k%2B1}_{i1},...,r^{k%2B1}_{ib}]"> and transition probabilities <img src="https://render.githubusercontent.com/render/math?math=[\pi^{k%2B1}_{i1},...,\pi^{k%2B1}_{ib}]"> from the output, for each parent node <img src="https://render.githubusercontent.com/render/math?math=i">
				* create a child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}"> and assign its hidden state as the set of all resulting hidden states from applying action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> to all parent nodes
				* for each node <img src="https://render.githubusercontent.com/render/math?math=j"> in the child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}">, update their transition probability so that it represents the probability of transitioning to the state represented by this node after applying the corresponding actions to the state of the root node: <img src="https://render.githubusercontent.com/render/math?math=\pi^{k%2B1}_j \leftarrow \pi^{k%2B1}_j \pi^k_{\rho(j)}">, where <img src="https://render.githubusercontent.com/render/math?math=\rho(j)"> is the parent node of node <img src="https://render.githubusercontent.com/render/math?math=j">
				* assign the transition reward of the child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=r_{N^{k%2B1}} = \sum_{j \in N^{k%2B1}} \pi_j^{k%2B1} r_j^{k%2B1}">
				* for each node <img src="https://render.githubusercontent.com/render/math?math=j"> in the child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}">, pass their resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}_j"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{k%2B1}_j) \rightarrow p^{k%2B1}_j,v^{k%2B1}_j">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^{k%2B+1}_j"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{k%2B1}_j"> from the output
				* assign the policy distribution over actions of the child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=p_{N^{k%2B1}} = [ \sum_{j \in N^{k%2B1}} \pi_j^{k%2B1} p_{j1}^{k%2B1} , ... , \sum_{j \in N^{k%2B1}} \pi_j^{k%2B1} p_{ja}^{k%2B1} ]">
				* assign the value of the child node set <img src="https://render.githubusercontent.com/render/math?math=N^{k%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=v_{N^{k%2B1}} = \sum_{j \in N^{k%2B1}} \pi_j^{k%2B1} v_j^{k%2B1}">
				* backpropagate the predicted value <img src="https://render.githubusercontent.com/render/math?math=v_{N^{k%2B1}}"> up the search path
		* sample an action based on the visit count of each child node of the root node
		* apply the sampled action to the environment and observe the resulting transition reward
	* once the episode is over, save the game trajectory (including the MCTS results) into the replay buffer
	* sample a number of game trajectories from the replay buffer:
		* pass the first observation of the environment <img src="https://render.githubusercontent.com/render/math?math=o_0"> from the game trajectory to the representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_0) \rightarrow s^0"> and get the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> from the output
		* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^0) \rightarrow p^0,v^0">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^0"> and value <img src="https://render.githubusercontent.com/render/math?math=v^0"> from the output
		* for every time step <img src="https://render.githubusercontent.com/render/math?math=t"> in the game trajectory:
			* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^t_i"> for each node <img src="https://render.githubusercontent.com/render/math?math=i"> in the current node set <img src="https://render.githubusercontent.com/render/math?math=N^t"> and the corresponding action <img src="https://render.githubusercontent.com/render/math?math=a^{t%2B1}"> into the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^t_i,a^{t%2B1}) \rightarrow [s^{t%2B1}_{i1},...,s^{t%2B1}_{ib}],[r^{t%2B1}_{i1},...,r^{tk%2B1}_{ib}],[\pi^{t%2B1}_{i1},...,\pi^{t%2B1}_{ib}]">, and get the predicted resulting hidden state representations <img src="https://render.githubusercontent.com/render/math?math=[s^{t%2B1}_{i1},...,s^{t%2B1}_{ib}]">, transition rewards <img src="https://render.githubusercontent.com/render/math?math=[r^{t%2B1}_{i1},...,r^{t%2B1}_{ib}]"> and transition probabilities <img src="https://render.githubusercontent.com/render/math?math=[\pi^{t%2B1}_{i1},...,\pi^{t%2B1}_{ib}]"> from the output, for each node <img src="https://render.githubusercontent.com/render/math?math=i">
			* create a node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}"> and assign its hidden state as the set of all resulting hidden states from applying action <img src="https://render.githubusercontent.com/render/math?math=a^{t%2B1}"> to all nodes of the node set <img src="https://render.githubusercontent.com/render/math?math=N^t"> from the previous time step
			* for each node <img src="https://render.githubusercontent.com/render/math?math=j"> in the node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}">, update their transition probability so that it represents the probability of transitioning to the state represented by this node after applying the actions taken in this game trajectory so far to the initial state of this game trajectory: <img src="https://render.githubusercontent.com/render/math?math=\pi^{t%2B1}_j \leftarrow \pi^{t%2B1}_j \pi^t_{\rho(j)}">, where <img src="https://render.githubusercontent.com/render/math?math=\rho(j)"> is the corresponding node from the previous time step of node <img src="https://render.githubusercontent.com/render/math?math=j">
			* assign the transition reward of the node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=r_{N^{t%2B1}} = \sum_{j \in N^{t%2B1}} \pi_j^{t%2B1} r_j^{t%2B1}">
				* this predicted transition reward <img src="https://render.githubusercontent.com/render/math?math=r_{N^{t%2B1}}"> is matched to the actual transition reward target received from the environment
			* for each node <img src="https://render.githubusercontent.com/render/math?math=j"> in the node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}">, pass their resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{t%2B1}_j"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{t%2B1}_j) \rightarrow p^{t%2B1}_j,v^{t%2B1}_j">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^{t%2B+1}_j"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{t%2B1}_j"> from the output
			* assign the policy distribution over actions of the node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=p_{N^{t%2B1}} = [ \sum_{j \in N^{t%2B1}} \pi_j^{t%2B1} p_{j1}^{t%2B1} , ... , \sum_{j \in N^{t%2B1}} \pi_j^{t%2B1} p_{ja}^{t%2B1} ]">
				* this predicted policy distribution <img src="https://render.githubusercontent.com/render/math?math=p_{N^{t%2B1}}"> is matched to the child node visit count distribution outputted by MCTS at that time step in that game trajectory
			* assign the value of the node set <img src="https://render.githubusercontent.com/render/math?math=N^{t%2B1}"> as: <img src="https://render.githubusercontent.com/render/math?math=v_{N^{t%2B1}} = \sum_{j \in N^{t%2B1}} \pi_j^{t%2B1} v_j^{t%2B1}">
				* this predicted value <img src="https://render.githubusercontent.com/render/math?math=v_{N^{t%2B1}}"> is matched to the value outputted by MCTS at that time step in that game trajectory
			* update the weights of the representation, dynamics and prediction function based on these three targets

### Implementation details

Rather than a list of nodes, the node set is implemented as a single class object; the same as a node in regular MCTS. Comparisons between how the attributes are implemented between a node and node set are detailed in the table below:

| Attribute | Node  | Node Set |
| ------------- | ------------- | ------------- |
| Prior  | A scalar value obtained from the policy value of the parent node; used to calculate UCB score | A scalar value obtained from the policy value of the parent node set; used to calculate UCB score |
| Hidden State Representation  | A <img src="https://render.githubusercontent.com/render/math?math=d">-length vector representation of the state represented by this node (where <img src="https://render.githubusercontent.com/render/math?math=d"> is determined by a hyperparameter) | An <img src="https://render.githubusercontent.com/render/math?math=n \times d"> matrix of vector representations for each stochastic state represented by this node set (where <img src="https://render.githubusercontent.com/render/math?math=n"> is the number of nodes in this node set) |
| Transition Probabilities | The Node class does not have this property, but if it were implemented, this value would be 1 | An <img src="https://render.githubusercontent.com/render/math?math=n">-length vector that denotes the probability of reaching each node state in the node set, if we were to apply the corresponding actions to the initial state |
| Transition Reward | A scalar value obtained from the dynamics function, denoting the predicted reward received for transitioning to the state represented by this node, from the  state represented by the parent node | A scalar value obtained by taking the expected value of the transition rewards from each node in the node set, denoting the predicted reward received for transitioning to the stochastic state represented by this node set, from the stochastic state represented by the parent node set; we don't need to keep track of the individual transition rewards as only the expected transition reward will be used to match against the target during training |
| Policy | An <img src="https://render.githubusercontent.com/render/math?math=a">-length vector obtained from the prediction function, denoting the prior action distribution for the child nodes |  An <img src="https://render.githubusercontent.com/render/math?math=a">-length vector obtained by taking the expected value of the prior action distributions from each node in the node set, denoting the prior action distribution for the child node sets; we don’t need to keep track of the individual prior action distributions as only the expected prior action distirbution will be used to match against the target during training |
| Value | A scalar value obtained from the prediction function, denoting the predicted return received if we were to follow the agent's policy at the state represented by this node for the remainder of the episode | A scalar value obtained by taking the expected value of the values from each node in the node set, denoting the predicted return received if we were to follow the agent's policy at the stochastic state represented by this node set for the remainder of the episode; we don’t need to keep track of the individual values as only the expected value will be used to match against the target during training |
| Children | A list of child nodes for each corresponding action in the action space | A list of child node sets for each corresponding action in the action space |
| Cumulative Value | A scalar value that accumulates values backpropagated up the search path; the average value is used to calculate UCB score | A scalar value that accumulates values backpropagated up the search path; the average value is used to calculate UCB score |
| Number of Visits | The number of times this node has been visited during MCTS; dividing the cumulative value by the visit count gives the average value | The number of times this node set has been visited during MCTS; dividing the cumulative value by the visit count gives the average value |

## Environment

## Results and Discussion
* analyze runtime and space complexity with respect to the branching factor hyperparameter

## Comparisons
* greater power to describe stochastic states, at the cost of greater memory requirements
* do we need to use exactly the same stochastic branching factor as the environment? Probably not
	* Even using a stochastic branching factor of 2 may improve performance compared to 1; might make it easier for hidden state representation to represent stochastic states

## Future Work
* merge similar resulting states together
* disregard low probability resulting states
* modify ucb score to factor in depth level of node to control for branching factor exploding

## MuZero Technical Details
Below is a description of how the MuZero algorithm works in more detail.

### Data structures
MuZero is comprised of three neural networks: 
* A representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_t) \rightarrow s^0">, which given an observation <img src="https://render.githubusercontent.com/render/math?math=o_t"> from the environment at time step <img src="https://render.githubusercontent.com/render/math?math=t">, outputs the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> of the observation at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=0"> (this hidden state will be used as the root node in MCTS, so its hypothetical time step is zero)
	* The representation function is used in tandem with the dynamics function to represent the environment's state in whatever way the algorithm finds useful in order to make accurate predictions for the reward, value and policy
* A dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^k,a^{k%2B1}) \rightarrow s^{k%2B1},r^{k%2B1}">, which given a hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^k"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k"> and action <img src="https://render.githubusercontent.com/render/math?math=a^{k%2B1}"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k%2B1">, outputs the predicted resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}"> and transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{k%2B1}"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k%2B1">
	* The dynamics function is the learned transition model, which allows MuZero to utilize MCTS and plan hypothetical future actions on future board states
* A prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^k) \rightarrow p^k,v^k">, which given a hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^k">, outputs the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^k"> and value <img src="https://render.githubusercontent.com/render/math?math=v^k"> at hypothetical time step <img src="https://render.githubusercontent.com/render/math?math=k">
	* The prediction function is used to limit the search breadth by using the policy output to prioritize MCTS to search for more promising actions, and limit the search depth by using the value output as a substitute for a Monte Carlo rollout

A replay buffer is used to store the history of played games, and will be sampled from during training.

### Self-play
At every time step during self-play, the environment's current state is passed into MuZero's representation function, which outputs the hidden state representation of the current state. Monte Carlo Tree Search is then performed for a number of simulations specified in the config parameter. 

In each simulation of MCTS, we start at the root node and traverse the tree until a leaf node (non-expanded node) is selected. Selection of nodes is based on a modified [UCB score](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation) that is dependent on the mean action Q-value and prior probability given by the prediction function (more detail can be found in Appendix B of the [MuZero](https://arxiv.org/pdf/1911.08265.pdf) paper). The mean action Q-value is min-max normalized to account for environments where the value is unbounded.

The leaf node is then expanded by passing the parent node's hidden state representation and the corresponding action into the dynamics function, which outputs the hidden state representation and transition reward for the leaf node.

The leaf node's hidden state representation is then passed into the prediction function, which outputs a policy distribution that serves as the prior probability for the leaf node's child nodes, and a value which is meant to be the predicted value of a "Monte Carlo rollout".

Finally, this predicted value is backpropagated up the tree, resulting in all nodes in the search path of the current simulation updating their mean action Q-values. The min and max values used in min-max normalization are updated if any of the nodes in the search path have new mean action Q-values that exceed the min-max bounds.

![Alt text](assets/muzero_mcts1.PNG)

Once the simulations are finished, an action is sampled from the distribution of visit counts of every child node of the root node. A temperature parameter controls the level of exploration when sampling actions. Set initially high to encourage exploration, the temperature is gradually reduced throughout self-play to eventually make action selection more greedy. The action selected is then executed in the environment and MCTS is conducted on the environment's next state until termination.

![Alt text](assets/muzero_mcts2.PNG)

### Training
At the end of every game of self-play, MuZero adds the game history to the replay buffer and samples a batch to train on. The game history contains the state, action and reward history of the game, as well as the MCTS policy and value results for each time step.

For each game, a random position is sampled and is unrolled a certain amount of timesteps specified in the config parameter. The sampled position is passed into the representation function to get the hidden state representation. For each unrolled timestep, the corresponding action taken during the actual game of self-play is passed into the dynamics function, along with the current hidden state representation. In addition, each hidden state representation is passed into the prediction function to get the corresponding predicted policy and value for each timestep.

The predicted rewards outputted by the dynamics function are matched against the actual transition rewards received during the game of self-play. The predicted policies outputted by the prediction function are matched against the policies outputted by the MCTS search. 

The "ground truth" for the value is calculated using <img src="https://render.githubusercontent.com/render/math?math=n">-step bootstrapping, where <img src="https://render.githubusercontent.com/render/math?math=n"> is specified in the config parameter. If <img src="https://render.githubusercontent.com/render/math?math=n"> is a number larger than the episode length, then the value is calculated using the actual discounted transition rewards of the game of self-play, and reduces to the Monte Carlo return. If <img src="https://render.githubusercontent.com/render/math?math=n"> is less than or equal to the episode length, then the discounted transition rewards are used until the <img src="https://render.githubusercontent.com/render/math?math=n">-step, at which point the value outputted by the MCTS search (i.e. the mean action Q-value of the root node) is used to bootstrap. The predicted values outputted by the prediction function are then matched against these calculated values.

The three neural networks are then trained end-to-end, matching the predicted rewards, values and policies with the "ground truth" rewards, values and policies. L2 regularization is used as well.

![Alt text](assets/muzero_train.PNG)

(MuZero diagrams can be found on page 3 of their [paper](https://arxiv.org/pdf/1911.08265.pdf))

## File Descriptions
* `classes.py` holds data structure classes used by MuZero
* `main.py` holds functions for self-play, MCTS, training and testing
	* `self_play` is the main function to call; it initiates self-play and trains MuZero
* `models/` holds saved neural network models used by MuZero
* `replay_buffers/` holds replay buffer instances, saved during self-play
* `recordings/` holds video file recordings of game renders when testing MuZero
* `assets/` holds media files used in this `README.md`
* `requirements.txt` holds all required dependencies, which can be installed by typing `pip install -r requirements.txt` in the command line

For this project, I'm using Python 3.7.4.

## Additional Resources
* [Full interview with David Silver, who led the AlphaGo team](https://www.youtube.com/watch?v=uPUEq8d73JI)
* [DeepMind AlphaGo webpage](https://deepmind.com/research/case-studies/alphago-the-story-so-far)
* [DeepMind MuZero webpage](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules)
* [Playlist of Youtube videos related to AlphaGo](https://www.youtube.com/playlist?list=PLqYmG7hTraZBy7J_4ynYPc0Ml1RUGcLmD)
* [A Youtube video describing an overview of the MuZero algorithm](https://www.youtube.com/watch?v=szbvm8aNDxw)
* Link to MuZero pseudocode ([v1](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py) and [v2](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py))