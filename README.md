# SigmaZero
This is a repo where I generalize DeepMind's MuZero reinforcement learning algorithm on stochastic environments, and create an algorithm I call SigmaZero (stochastic MuZero). For more details on the algorithm specifics, check out the original [paper](https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D) and my [project](https://github.com/chiamp/muzero-cartpole) on applying the MuZero algorithm on the cartpole environment.

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
				* pass the resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{k%2B1}"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{k%2B1}) \rightarrow p^{k%2B1},v^{k%2B1}">, and get the predicted policy over actions <img src="https://render.githubusercontent.com/render/math?math=p^{k%2B+1}"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{k%2B1}"> from the output
				* backpropagate the predicted value <img src="https://render.githubusercontent.com/render/math?math=v^{k%2B1}"> up the search path
		* sample an action based on the visit count of each child node of the root node
		* apply the sampled action to the environment and observe the resulting transition reward
	* once the episode is over, save the game trajectory (including the MCTS results) into the replay buffer
	* sample a number of game trajectories from the replay buffer:
		* pass the first observation of the environment <img src="https://render.githubusercontent.com/render/math?math=o_t"> from the game trajectory to the representation function, <img src="https://render.githubusercontent.com/render/math?math=h(o_t) \rightarrow s^0"> and get the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> from the output
		* pass the hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^0"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^0) \rightarrow p^0,v^0">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^0"> and value <img src="https://render.githubusercontent.com/render/math?math=v^0"> from the output
		* for every time step <img src="https://render.githubusercontent.com/render/math?math=t"> in the game trajectory:
			* pass the corresponding action <img src="https://render.githubusercontent.com/render/math?math=a^{t%2B1}"> and current hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^t"> to the dynamics function, <img src="https://render.githubusercontent.com/render/math?math=g(s^t,a^{t%2B1}) \rightarrow s^{t%2B1},r^{t%2B1}"> to get the predicted resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{t%2B1}"> and transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{t%2B1}"> from the output
				* this predicted transition reward <img src="https://render.githubusercontent.com/render/math?math=r^{t%2B1}"> is matched to the actual transition reward target received from the environment
			* pass the resulting hidden state representation <img src="https://render.githubusercontent.com/render/math?math=s^{t%2B1}"> into the prediction function, <img src="https://render.githubusercontent.com/render/math?math=f(s^{t%2B1}) \rightarrow p^{t%2B1},v^{t%2B1}">, and get the predicted policy distribution over actions <img src="https://render.githubusercontent.com/render/math?math=p^{t%2B1}"> and value <img src="https://render.githubusercontent.com/render/math?math=v^{t%2B1}"> from the output
				* this predicted policy distribution <img src="https://render.githubusercontent.com/render/math?math=p^{t%2B1}"> is matched to the child node visit count distribution outputted by MCTS at that time step in that game trajectory
				* this predicted value <img src="https://render.githubusercontent.com/render/math?math=v^{t%2B1}"> is matched to the value outputted by MCTS at that time step in that game trajectory
			* update the weights of the representation, dynamics and prediction function based on these three targets

## SigmaZero
* SigmaZero (Stochastic MuZero)

## Environment

## Results

## Comparisons
* greater power to describe stochastic states, at the cost of greater memory requirements
* do we need to use exactly the same stochastic branching factor as the environment? Probably not
	* Even using a stochastic branching factor of 2 may improve performance compared to 1; might make it easier for hidden state representation to represent stochastic states

## Future Work
* merge similar resulting states together
* disregard low probability resulting states

## Thoughts
In our effort to make machines more intelligent, we tried to encode our knowledge and expertise of the domain we're working with into the machine. Deep Blue is an example of that; it is a monumental feat of engineering and collaboration between computer scientists and chess grandmasters, working together and pooling their knowledge and experience to create a machine capable of superhuman performance in chess. 

Yet, this approach has its limits. For one, the time and resources required to figure out what domain-specific knowledge would be useful for the machine to have, and then to figure out how to quantify and hard-code this knowledge into a computer system are huge. Secondly, the knowledge we encode into the machine is domain-specific, meaning that encoding a computer system with chess-specific knowledge, will make the computer only good at playing chess. If we wanted to make a machine capable of playing Go at a high level, we would need to find professional Go players and redo the whole process of figuring out what sort of Go knowledge we think is useful to put into the computer system. This process has to be repeated for every unique domain we'd like to create an AI system for, which takes a lot of time and resources. Not to mention the fact that there may be domains that such experts don't exist currently, or the nature of the domain itself makes it hard to quantify a hard-coded evaluation function.

With the advent of machine learning, we can now take a different approach to making machines more intelligent. Rather than telling a machine explicitly what to look out for, and what we consider to be good or bad, we can use neural networks to have the machine learn these values for itself through its own experiences of self-play. AlphaGo was the first step in this direction; because of the nature of the game, coming up with a hard-coded evaluation function for Go was much more difficult compared to chess. So instead, DeepMind decided to let the machine learn for itself. After studying many human games and playing against itself over and over again, it was able to figure out what board positions in Go were considered good or bad, via its own experience. Rather than being provided a hard-coded evaluation function, AlphaGo learned its own evaluation function based on its own experience, and using that, was able to defeat Lee Sedol.

AlphaGo Zero removed the human element; it did not see any human Go games when training, and only learned through self-play. The fact that it was able to defeat AlphaGo 100 games to 0, perhaps hints at the idea that there are limits to knowledge solely derived from humans. Human knowledge of Go is deep; the game has been around for thousands of years, and the theory and strategy has been well developed and studied. Yet the computer program that was devoid of any human knowledge was able to defeat the computer program that was partially trained on human games, 100 games to 0. It seems that the knowledge that we humans have gathered, may not be the most optimal, rather specifically in this example, our human bias served to hinder AlphaGo's performance. Indeed, we find in the research paper that AlphaGo Zero discovered new variations of well-established pattern sequences that humans considered optimal in Go. And now these novel AI discoveries are in turn being studied by professional Go players (mentioned in an [interview](https://youtu.be/uPUEq8d73JI?t=5489) by David Silver, who lead the AlphaGo team). This idea is what makes me very excited as to what other novel things AI algorithms can show us...what other things have we accepted as "optimal", that can actually be improved upon, based on what AI algorithms can discover?

And it's not just in Go. AlphaZero shows us that the same algorithm used to achieve superhuman performance in Go, can also be applied to different domains in chess and shogi. No longer do we have to spend extra time and resources to hard-code domain-specific knowledge into the AI's evaluation functions, when we can use a general reinforcement learning algorithm to learn the evaluation function via self-play. This is extremely exciting, especially for problems in challenging domains where hard-coded evaluation functions are difficult to quantify; for example, in the field of autonomous driving and robots. The fact that AlphaZero was able to exceed Stockfish's performance in chess, considering the fact that Stockfish, at the time, also used hard-coded evaluation functions similar to Deep Blue, gives further credence to the value of this type of reinforcement learning algorithm. 

The elimination of the need to provide a transition model, is another step toward the generalization of this reinforcement learning algorithm. It may seem trivial in the space of board games, where the rules and consequences of actions are well-defined. But for real-world applications, we would need a transition model that mimics the real world, in order to allow the machine to accurately learn and plan future actions accordingly. Unfortunately, the real world isn't as discrete and deterministic as the world of board games; it can be continuous and stochastic, and so the "rules of the world" are not as well-defined. 

Yet humans are able to navigate through this messy world with relative success. Indeed humans have learned a transition model of the world themselves through experience; for example, we can reasonably predict the trajectory of a ball thrown to us to catch it, we know that an object can fall and break if we push it off the table, we can predict how slippery a surface would be to walk on just by looking at it etc. By using our "transition model", i.e. our understanding of "the rules of the game", we can plan accordingly and take actions to achieve our goals. We learn this "transition model" through our own experiences in life, and that's exactly what's happening with MuZero as well. Without being supplied the "rules of the game", it builds a transition model for itself via self-play and using that, plans for hypothetical future situations with MCTS and takes actions to maximize its reward. With this algorithm, we're one step closer towards using artificial intelligence in real world applications.

Artificial intelligence has the potential to solve a lot of problems for us. The fact that it could potentially discover novel things that humans either prematurely dismissed as non-optimal, or never even considered, is exciting to me, especially in the fields of genetics, medicine and energy. I think we're at an exciting period of time where AI technology is expanding at an exponential rate, and I can't wait to see what the future has in store for us!

## What is gym?
[gym](https://gym.openai.com/envs/) is a suite of virtual environments provided by OpenAI, to test reinforcement learning algorithms on. The suite contains everything from simple text games, to retro Atari games, to even 3D physics simulators.

For this project, I apply the MuZero algorithm to the cart pole environment. The goal of the agent is to balance a pole on a cart, by moving the cart left and right. The agent is incentivized with a reward equal to how long the pole stays balanced on the cart; the longer the pole is kept balanced, the bigger the reward.

Below you can see the progression of the agent learning over time. Initially, it knows nothing of the environment and performs terribly. But gradually through experience, the agent learns "the rules of the game" and figures out an optimal strategy to balance the pole on top of the cart indefinitely.

#### Initial (0 games played)
![Alt text](assets/cartpole_0_games.gif)

#### After playing 100 games
![Alt text](assets/cartpole_100_games.gif)

#### After playing 200 games
![Alt text](assets/cartpole_200_games.gif)

#### After playing 300 games
![Alt text](assets/cartpole_300_games.gif)

#### After playing 400 games
![Alt text](assets/cartpole_400_games.gif)

#### After playing 500 games
![Alt text](assets/cartpole_500_games.gif)

#### After playing 600 games
![Alt text](assets/cartpole_600_games.gif)

#### After playing 700 games
![Alt text](assets/cartpole_700_games.gif)

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