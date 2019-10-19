# MazeExperiment

### A simple project testing memory of LSTM networks.

Using Proximal Policy Optimization, agents are trained to pass randomly generated mazes. Each maze is guaranteed to contain no cycles and has a single starting point and signle goal. Each agent observes the same maze 10 times, so to efficiently solve the task, it needs to remember the path leading to the goal to not waste time in the following runs.

`python3 visualise_play.py` shows visualisation of the agent solving mazes.
`python3 maze_trainer.py` starts training of the agent.

The project requires tensorflow 1.11 or higher and pygame (for visualisation). 
