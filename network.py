import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def normalize_loss(loss):
    return loss/(tf.abs(tf.stop_gradient(loss))+0.5)
    

class AC_Network():
    def __init__(self,scope,trainer,batch_size,map_len):
        self.batch_size = batch_size
        self.map_len = map_len
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,self.map_len*self.map_len+4],dtype=tf.float32)
            self.batchsize = tf.placeholder(shape=[1],dtype=tf.int32)
            self.sequence_lengths = tf.placeholder(shape=[None],dtype=tf.int32)
            
        
            fc1 = slim.fully_connected(self.inputs,64,
                activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #LSTM BLOCK
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256)
            c_init = np.zeros([1]+[lstm_cell.state_size.c], np.float32)
            #print(c_init.shape)
            h_init = np.zeros([1]+[lstm_cell.state_size.c], np.float32)
            self.state_init = [c_init, h_init]
            
            c_init = np.zeros([self.batch_size]+[lstm_cell.state_size.c], np.float32)
            #print(c_init.shape)
            h_init = np.zeros([self.batch_size]+[lstm_cell.state_size.c], np.float32)
            self.batch_init = [c_init, h_init]
            print([-1]+[lstm_cell.state_size.c])
            c_in = tf.placeholder(tf.float32, [None]+[lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [None]+[lstm_cell.state_size.c])
            self.state_in = (c_in, h_in)
            
            rnn_in = tf.reshape(fc1,[self.batchsize[0],-1,64])

            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=self.sequence_lengths,swap_memory=True,
                time_major=False)
            
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:,:], lstm_h[:,:])
            
            rnn_out = tf.reshape(lstm_outputs, [-1,256])
            
            fc2 = slim.fully_connected(rnn_out,64,
                activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            #Output layers for policy and value estimations
            action = slim.fully_connected(fc2,4,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            self.policy = action
            
            self.result = tf.multinomial(tf.log(action),1)
            
            self.value = slim.fully_connected(fc2,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.reward = slim.fully_connected(fc2,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.001),
                biases_initializer=None)
            #Only the global network need ops for loss functions and gradient updating.
            if scope == 'global':
                self.actions = tf.placeholder(shape=[None,1],dtype=tf.int32)
                self.action_onehot = tf.one_hot(self.actions[:,0],4,dtype=tf.float32)
                
                self.old_probabilities = tf.placeholder(shape=[None],dtype=tf.float32)
                
                
                act = self.policy
                
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.target_r = tf.placeholder(shape=[None],dtype=tf.float32)
                self.masks = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                def re(x,y):
                    print(x.shape)
                    print(y.shape)
                    out = tf.reduce_sum(x*y,[1])
                    print(out.shape)
                    return out
                
                self.responsible_outputs = re(self.action_onehot,act)

                #Loss functions
                self.reward_loss = 0.5*tf.to_float(tf.shape(self.masks)[0])/(tf.to_float(tf.reduce_sum(self.masks))+0.001)*tf.reduce_sum(self.masks*(tf.square(self.target_r - tf.reshape(self.reward,[-1]))))
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(act * tf.log(act+1e-17))
                self.ratios = (self.responsible_outputs+1e-17)/(self.old_probabilities+1e-17)
                self.policy_loss = -tf.reduce_sum(tf.minimum(self.ratios*self.advantages,tf.clip_by_value(self.ratios,1.0-0.1,1.0+0.1)*self.advantages))
                self.loss = 0.5 * normalize_loss(self.value_loss) + normalize_loss(self.policy_loss) - normalize_loss(self.entropy) * 0.0005 + 0.1*normalize_loss(self.reward_loss)

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,100.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                


