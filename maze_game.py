import numpy as np
from random import randrange, sample,random
from queue import Queue, PriorityQueue

def generate_lab(n):

    lab = np.zeros([n*2-1,n*2-1])

    start = (randrange(n)*2,randrange(n)*2)

    to_consider = PriorityQueue()

    to_consider.put((random(),(start,start)))

    #lab[start[0],start[1]] = 1

    while not to_consider.empty():
        _,position = to_consider.get()
        
        a,b = position[0]
        #print(a,b)
        if lab[a,b] == 1:
            continue
        
        source_a, source_b = position[1]
        
        lab[a,b] = 1
        lab[(a+source_a)//2,(b+source_b)//2]=1
        
        for x,y in [(2,0),(0,2),(-2,0),(0,-2)]:
            if -1< a+x <2*n-1 and -1< b+y <2*n-1 and lab[a+x,b+y]==0:
                to_consider.put((random(),((a+x,b+y),(a,b))))
    
    '''
    for i in range(2*n-1):
        x = ""
        for j in range(2*n-1):
            if lab[i,j] ==0:
                x+= "#"
            else:
                x+= " "
        print(x)
       ''' 
    return lab

class maze_game:
    
    def __init__(self,map_size,observation_size,episodes_count,step_count):
        self.map_size = map_size
        self.observation_size = observation_size
        self.episodes = episodes_count
        self.steps = step_count
        self.action_space = [(1,0),(0,1),(-1,0),(0,-1)]
        
        
    def generate_observation(self):
        obs_size = self.observation_size
        squared = obs_size*obs_size
        obs = np.zeros([squared+4])
        
        u,v = self.player_pos
        
        obs[0:squared] = self.draw_lab[u:u+obs_size,v:v+obs_size].flatten()
        #print(half_obs)
        
        '''
        
        for x in range(self.observation_size):
            #show = ""
            for y in range(self.observation_size):
                a = u+x-(self.observation_size//2)
                b = v+y-(self.observation_size//2)
                if -1< a <self.map_size*2-1 and -1< b <self.map_size*2-1:
                    #show+= str(int(self.lab[a,b]))
                    obs[x*self.observation_size+y] = self.lab[a,b]
                #else:
                    #show+="0"
            #print(show)
        
        '''
        
        if not self.last_action == -1:
            obs[squared+self.last_action] = 1 
                
        return obs
        
    
    def compute_minimum_path(self):
        visited = set()
        #visited.add(self.start)
        
        q = Queue()
        q.put((self.start,0))
        
        
        while not q.empty():
            x = q.get()
            #print(x)
            pos, dist = x
            if pos == self.end:
                return dist
            if pos in visited:
                continue
            x,y = pos
            
            for dx,dy in self.action_space:
                a = x+dx
                b = y+dy
                
                if -1< a <self.map_size*2-1 and -1< b <self.map_size*2-1 and not self.lab[a,b] == 0 and not (a,b) in visited:
                    q.put(((a,b),dist+1))
            
            visited.add(pos)
    
    def reset(self):
        self.lab = generate_lab(self.map_size)
        
        
        self.start = (randrange(self.map_size)*2,randrange(self.map_size)*2)
        self.end = self.start
        
        while self.end == self.start:
            self.end = (randrange(self.map_size)*2,randrange(self.map_size)*2)
        
        self.lab[self.end[0],self.end[1]] = 2
        self.draw_lab = np.pad(self.lab,self.observation_size//2,'constant')
        #self.lab[self.start[0],self.start[1]] = 3
        #print(self.lab)
        self.minimum_path = self.compute_minimum_path()
        #print(self.minimum_path)
        self.last_action = -1
        self.player_pos = self.start
        
        self.episode_steps = 0
        self.episodes_count = 1
        self.score_per_episode = []
        
        return self.generate_observation(), 0, False
        
        
    def step(self, action):
        
        
        self.last_action = action
        dx, dy = self.action_space[action]
        
        a = self.player_pos[0]+dx
        b = self.player_pos[1]+dy
        
        if -1< a <self.map_size*2-1 and -1< b <self.map_size*2-1 and not self.lab[a,b] == 0:
            self.player_pos = (a,b)
        
        reward = 0
        
        self.episode_steps+=1
        
        if self.lab[self.player_pos[0],self.player_pos[1]]==2:
            reward = (self.minimum_path/self.episode_steps)
            self.episodes_count += 1
            self.episode_steps = 0
            self.player_pos = self.start
            self.score_per_episode.append(reward)
        
        done = False
        
        if self.steps < self.episode_steps:
            self.episodes_count+=1
            self.player_pos = self.start
            self.episode_steps = 0
            self.score_per_episode.append(reward)
        
        statistics = None
        
        if self.episodes < self.episodes_count:
            done = True
            statistics = self.score_per_episode
        
        
        
        return self.generate_observation(),reward,done,statistics


if __name__ == '__main__':
    
    maze = maze_game(4,5,1000,100)

    #print(maze.reset())
    for _ in range(100):
        maze.reset()
        for i in range(1000):
            
            x = maze.step(randrange(4))[0][:25]
            #print(np.reshape(x,[5,5]))
                
#plain maze:42735,042735043
#full: 11091,854419411
