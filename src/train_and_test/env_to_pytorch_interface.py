###############
## Prepare data set
##################
import torch
import string
import numpy as np
import random
import pandas as pd

print("Load env-to-pytorch interface .... ")



##############################
## CREATE TASK VECTOR --- modularized
#################################

def create_task_vector(env):
  
    task_vector = torch.zeros(env.task_vector_size) #5
    object_vector = torch.zeros(env.object_vector_size) #5
    quant_vector = torch.zeros(env.quant_vector_size) #11. 
    
    task_vector[env.task_n] = 1
    object_vector[env.object_n] = 1
    # for s_i in str(env.quant_n):
    #   if(s_i=='0'):
    #     quant_vector[10] = 1
    #   else:
    #     quant_vector[int(s_i)] = 1
    # if(env.quant_n_2 > -1):
    #   quant_vector[env.quant_n_2] = 1

    if(env.IsDecimal): # show digits sequentially in the beginning
      n_string = int2base(env.quant_n, env.base)
      if(env.leading_zero):
        n_string = "{:02d}".format(int(int2base(env.quant_n, env.base)))
      n_str_length = len(n_string)

      if(env.time<n_str_length):
        s_i = n_string[env.time]
        if(s_i=='0'):
          quant_vector[env.base] = 1
        else:
          quant_vector[int(s_i)] = 1
        
        if(env.leading_zero):
          object_vector[-env.time-1] = 1



    else:
          quant_vector[env.quant_n] = 1

    stacked_vector = torch.cat([task_vector, quant_vector, object_vector])
    stacked_vector = stacked_vector.reshape([1, env.task_vector_length])
    return stacked_vector




### Higher level task
def create_task_data():
  env = CountEnv()

  epoch_duration = 200

  height = env.observation[:,0].size
  width = env.observation[0,:].size

  all_imgs = np.zeros([epoch_duration,height, width])
  all_actions = np.zeros([epoch_duration,env.n_actions])
    
  for t in range(epoch_duration):
    env.update("left")                       ## !!!!!!!! WHY is this here
    all_imgs[t,:,:] = env.observation/255
    all_actions[t,:] = env.action_onehot
  
  all_imgs = all_imgs.reshape(epoch_duration, 1, 1, height, width)
  all_actions = all_actions.reshape(epoch_duration, 1, env.n_actions)

  all_imgs = torch.from_numpy(all_imgs).float()
  all_actions = torch.from_numpy(all_actions).float()
  
  return all_imgs, all_actions


### just go from left to right
def create_epoche_imgs():
  env = CountEnv()

  epoch_duration = 200

  height = env.observation[:,0].size
  width = env.observation[0,:].size

  all_imgs = np.zeros([epoch_duration,height, width])
  all_actions = np.zeros([epoch_duration,env.n_actions])
    
  for t in range(epoch_duration):
    env.update("left")
    all_imgs[t,:,:] = env.observation/255.
    all_actions[t,:] = env.action_onehot
  
  all_imgs = all_imgs.reshape(epoch_duration, 1, 1, height, width)
  all_actions = all_actions.reshape(epoch_duration, 1, env.n_actions)

  all_imgs = torch.from_numpy(all_imgs).float()
  all_actions = torch.from_numpy(all_actions).float()
  
  return all_imgs, all_actions


def envImageAndActionToPytorchFormat(env):
    height = env.view_size
    width = env.view_size #env.observation[0,:].size

    img_hand = np.zeros([height, width])
    img_square = np.zeros([height, width])
    
    all_actions = np.zeros([env.n_actions])

    img_hand[:,:] = env.observation_hand/255.
    img_square[:,:] = env.observation_square/255.
    
    img_hand = torch.from_numpy(img_hand).float()
    img_square = torch.from_numpy(img_square).float()
    
    action = env.action_onehot
    img = torch.stack([img_hand,img_square])
    img = img.reshape(1, 2, height, width)
    action = action.reshape(1,env.n_actions)

    
    action = torch.from_numpy(action).float()

    return img, action

##############################
## ADD TASK LAYER --- modularized
#################################


def add_task_layer(image, img_size, env):
    
    ### Action: one-hot colomns determine - recite, touch, count, give
    #task_matrix = torch.zeros(img_size,img_size)
    #task_ones = torch.ones(img_size)
    #task_matrix[env.task_n, :] = task_ones
    #task_matrix = torch.reshape(task_matrix, [1,img_size,img_size])
    
    #### Object: one-hot colomns determine objects to count - square, events, nothing
    object_matrix = torch.zeros(img_size,img_size)
    object_ones = torch.ones(img_size)
    object_matrix[env.object_n, :] = object_ones
    object_matrix = torch.reshape(object_matrix, [1,img_size,img_size])
    
    #### Quantifier: All, 1, 2, 3, 4... : All uppest left corner, 1 right to it...
    quant_matrix = torch.zeros(img_size,img_size)
    col = env.quant_n%img_size
    row = int(env.quant_n/img_size )
    quant_matrix[row,col] = 1        
    quant_matrix = torch.reshape(quant_matrix, [1,img_size,img_size])    
    
    inp = image.reshape([2,img_size,img_size])  #.reshape([1,200,200])
    #stacked_image = torch.stack([inp[0],inp[1],task_matrix[0], object_matrix[0], quant_matrix[0] ]).reshape(1,5,img_size,img_size)
    
    stacked_image = image.reshape([1,2,img_size,img_size])
    
    return stacked_image
  
  




# #######
# # Create the following matrix:
# # All 1  2  3
# #  4  5  6  7
# #  8  9  10 11
# #  12 13 14 15

# quant_matrix = torch.zeros(4,4)

# for i in range(16):
#     col = i%4
#     row = int((i)/4 )
#     quant_matrix[row,col] = i


##############################
## ADD TASK LAYER
#################################


def add_task_layer_old(image, img_size, task_n = 0):
        
    task_matrix = torch.zeros(img_size,img_size)
    task_ones = torch.ones(img_size)
    task_matrix[task_n, :] = task_ones
    task_matrix = torch.reshape(task_matrix, [1,img_size,img_size])
    
    inp = image.reshape([2,img_size,img_size])  #.reshape([1,200,200])
    stacked_image = torch.stack([inp[0],inp[1],task_matrix[0] ]).reshape(1,3,img_size,img_size)
    
    return stacked_image
  
  
class result_tensor():
  
    def __init__(self):
        self.task = []
        self.n_obj = []
        self.episode = []
        self.accuracy = []
        
        self.losses = []
        self.losses_episodes = []
        self.runs = []
        
        self.n_one_to_ones = []
        self.n_right_number_words = []
        self.variabilities = []
                        
    def add_episode_result(self, task, n_obj, episode, accuracy, run_n,n_one_to_one=None, n_right_number_words=None, variabilities=None):
        self.task.append(task)
        self.n_obj.append(n_obj)
        self.episode.append(episode)
        self.accuracy.append(accuracy)
        self.runs.append(run_n)
        if(n_one_to_one is not None):
          self.n_one_to_ones.append(n_one_to_one)
        if(n_right_number_words is not None):
          self.n_right_number_words.append(n_right_number_words)
        if(variabilities is not None):
          self.variabilities.append(variabilities)        
    def add_loss(self, loss, episode): 
        self.losses.append(loss)
        self.losses_episodes.append(episode)
        
    def create_panda_df(self):
      
        normalized_variabilities = [float(i)/max(self.variabilities) for i in self.variabilities] 
        
        df = pd.DataFrame(
        {
            "task": self.task,
            "n_obj": self.n_obj,
            "episode": self.episode,
            "accuracy": self.accuracy,
            "losses": self.losses,
            "runs": self.runs,
            "n_one_to_ones": self.n_one_to_ones,
            "n_right_number_words": self.n_right_number_words,
            "variabilities": normalized_variabilities
        })
            
        return df

class DataSet():
  def __init__(self, epochs, task_vector_list):
    self.epochs = epochs
    self.task_vector_list = task_vector_list

def create_data_set(env, set_size):
    epochs = []
    env_task_list = []
    for n in range(set_size):
      env.reset()
      #print(env.task, env.n_squares)
      #if(env.task=="recite_n"):
      #print("--- in Creating batch: should be only 1 if recite_n")
      #print("env.task: ", env.task)
      #print("env.n_squares: ", env.n_squares)
      env.solve_task()
      #print("after solved task")

      task_vector = torch.zeros(env.task_vector_length)
      task_vector[env.task_n] = 1              
      task_vector = create_task_vector(env)

      env_task_list.append(task_vector)
      epochs.append(env.epoch)

    data_set = DataSet(epochs, env_task_list)

    return data_set



def get_batch_from_data_set(data_set, batch_size):

    env_img_list = []    # each list item represents one time instance, which contains a whole batch
    env_action_list = []
    env_relation_list = []
    n_list = []
    env_task_list = []
    episode_length_list = [0]*batch_size

    list1 = range(len(data_set.epochs))
    indicess = random.sample(list1, batch_size)
    #####
    #env_epochs = random.sample(data_set.epochs, batch_size)
    env_epochs = [data_set.epochs[i] for i in indicess]
    #env_task_list = [data_set.task_vector_list[i] for i in indicess]
    #env_task_list = torch.stack(env_task_list)

    for n in range(batch_size):
        env_epoch = env_epochs[n]
    
        for t in range(0, len(env_epoch) ): #(len(env.epoch)-1)
            episode_length_list[n] += 1
            stacked_img_coord = env_epoch[t]['img']            
            if(t>=len(env_img_list) ):
                env_img_list.append(stacked_img_coord)
                env_action_list.append(env_epoch[t]['action'])  
                env_task_list.append(env_epoch[t]['task_vector'])         
                n_list.append( np.array(n) )
            else:
                env_img_list[t] = torch.cat([env_img_list[t],  stacked_img_coord])
                env_action_list[t] = torch.cat([env_action_list[t],  env_epoch[t]['action']])
                env_task_list[t] = torch.cat([env_task_list[t],      env_epoch[t]['task_vector']])
                n_list[t] = np.append( n_list[t], n )
    # Dummy last entry: won't be needed in training, since after last step state_network not updated anymore
    n_list.append( np.array(n) )
    
    # Create index_list: list of indices which entry in batch are to be kept from one time step to the next
    current_ind_list = np.arange(batch_size)
    ind_list = []       
    for t in range(len(n_list) - 1 ):
        first_n = True
        for n_ in np.ndenumerate(n_list[t+1]):
            n = n_[1]
            appendy = np.where(n_list[t]==n)  
            if type(appendy) is tuple:
                appendy=appendy[0]
            if(first_n):
                ind_list.append( appendy  )
                first_n = False
            else:
                ind_list[t] = np.append(ind_list[t],  appendy  )
    
    for t in range(len(ind_list)):
          ind_list[t] =  torch.from_numpy( ind_list[t] ) 
    
    
       
    return  env_img_list, env_action_list, ind_list, episode_length_list, env_task_list



def create_batch(env, batch_size):

    env_img_list = []    # each list item represents one time instance, which contains a whole batch
    env_action_list = []
    env_relation_list = []
    n_list = []
    env_task_list = []
    task_vector_size = env.task_vector_length
    
    


    for n in range(batch_size):
        env.reset()
        
        env.solve_task()

        task_vector = torch.zeros(task_vector_size)
        task_vector[env.task_n] = 1
                
        task_vector = create_task_vector(env)
        env_task_list.append(task_vector)
      
        for t in range(0, len(env.epoch) ): #(len(env.epoch)-1)
            

            #stacked_img_coord = add_task_layer(env.epoch[t]['img'], env.img_size, env.task_n)
            #stacked_img_coord = add_task_layer(env.epoch[t]['img'], env.img_size, env)
            stacked_img_coord = env.epoch[t]['img']

            
            if(t>=len(env_img_list) ):
                env_img_list.append(stacked_img_coord)
                env_action_list.append(env.epoch[t]['action'])
                #env_relation_list.append(env.epoch[t+1]['rel'])
                
                n_list.append( np.array(n) )
            else:
                env_img_list[t] = torch.cat([env_img_list[t],  stacked_img_coord])
                env_action_list[t] = torch.cat([env_action_list[t],  env.epoch[t]['action']])
                #env_relation_list[t] = env_relation_list[t] #torch.cat([env_relation_list[t],  env.epoch[t+1]['rel']])    !!!!!!! #change
                
                #env_action_list[t] = torch.cat([env_action_list[t],  env.epoch[t+1]['action']])
                #print("ind_list[t]: ", ind_list[t])
                #print("np.where(current_ind_list==n)[0]", np.where(current_ind_list==n))
                n_list[t] = np.append( n_list[t], n )
    # Dummy last entry: won't be needed in training, since after last step state_network not updated anymore
    n_list.append( np.array(n) )
    
    current_ind_list = np.arange(batch_size)
    ind_list = []
        
    #print("n_list ", n_list )

    for t in range(len(n_list) - 1 ):
        #print("ind_list[t]: ", ind_list[t])
        first_n = True
        for n_ in np.ndenumerate(n_list[t+1]):
            n = n_[1]
            appendy = np.where(n_list[t]==n)  
            if type(appendy) is tuple:
                appendy=appendy[0]
            if(first_n):
                ind_list.append( appendy  )
                first_n = False
            else:
                ind_list[t] = np.append(ind_list[t],  appendy  )
            
            #pass
            #print(n)
            #if(np.any(current_ind_list[:, 0] == n))
            #ind_list[t] = current_ind_list[ind_list[t]]
            #current_ind_list = current_ind_list[ind_list[t]]
            
        
        #ind_list.append(np.where(current_ind_list==n)[0]  )
        #ind_list[t] = np.concatenate(ind_list[t], np.where(current_ind_list==n)[0])
    #print("ind_list: ", ind_list)
      
      
      
    for t in range(len(ind_list)):
          ind_list[t] =  torch.from_numpy( ind_list[t] ) 
    
    env_task_list = torch.stack(env_task_list)
    
    
      
    return  env_img_list, env_action_list, ind_list, env_relation_list, env_task_list  
  


def int2base(x, base):
    digs = string.digits + string.ascii_letters
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)



