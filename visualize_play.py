import numpy as np
import tensorflow as tf

from network import AC_Network
from maze_game import maze_game
from maze_visualiser import visualizer

from random import choice,choices,sample,randrange,random
from time import sleep
import os

map_len = 5

load_model = True
model_path = './model2'

tf.reset_default_graph()

if not os.path.exists(model_path):
    raise Exception("model folder not found")
    
with tf.device("/gpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int64,name='global_episodes',trainable=False)
    increment = global_episodes.assign_add(1)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network('global',trainer,1,map_len) # Generate global network
    num_workers = 8#multiprocessing.cpu_count() # Set workers to number of available CPU threads
    saver = tf.train.Saver(max_to_keep=5)
    
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    
    print ('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    print ('Loaded')
    
    network = master_network
    env = maze_game(12,map_len,10,300)
    vis = visualizer(env,400)
    rnn_state = network.state_init
    obs, reward, done = env.reset()
    last_obs = obs
    
    while True:
        
        
        lstm_c,lstm_h = rnn_state
        
        #print(lstm_c)
        a_dist,rnn_state,value,sampled = sess.run([network.policy,network.state_out,network.value,network.result], 
                                            feed_dict={network.inputs:[last_obs],
                                            network.state_in[0]:lstm_c,
                                            network.state_in[1]:lstm_h,
                                            network.batchsize:[1],
                                            network.sequence_lengths:([1])})
                                            
        
        action = sampled[0][0]#choices([0,1,2,3],weights=a_dist[i])[0]
        
        probability = a_dist[0][action]
        
        
        vis.draw()
        sleep(0.03)
        new_obs, reward, done, statistics = env.step(action)
        
        if done:
            rnn_state = network.state_init
            obs, reward, done = env.reset()
            last_obs = obs
        else:
            last_obs = new_obs

