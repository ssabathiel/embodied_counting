##########################################
###### SUPERVISED LEARNING
#####################################
#########################
################
##############
###########
##########
#########
########

import os
import torch
import torch.nn as nn
from environment.minimal_count_environment import *
from train_and_test.test_model import *
import datetime

print("Load training process....!!!")

def train_model(task_list=["touch_all_objects"], n_squares_ = 1, num_epochs = 2000, episode = 0, run_n = 0, model=None, params={}):

    directory_path_models = model.model_path + "Models/"
    if(not os.path.exists(directory_path_models) ):
      os.mkdir(directory_path_models)  
  
    model.train()   
    optimizer = torch.optim.Adam(model.parameters(), lr = model.lr) 

    torch.manual_seed(0)
    loss_fn = nn.MSELoss()
    
    max_epoch = num_epochs 
    test_every_n = 200
    test_every_n_initial = 100
    test_every_after = test_every_n
    test_rate_changes_after = 100
    update_rate = 0

    env = CountEnv(task = task_list[0], n_squares = n_squares_, display = "None", save_epoch = True, params=params )
    env.sample_task = True
    env.rand_n_squares=True
    env.task_list = task_list 
    episode = model.episode
    
    batch_size = params['batch_size'] if 'batch_size' in params else 16*len(task_list)    
    average_loss = 0
    old_average_loss = 5000000

    
    ## Initial Test before training   
    #test_env_on_every_task_and_N(env, model, episode, run_n, average_loss, test_every_n)
           
    ### Create Dataset
    print("Creating data set... ")
    data_set_size = params['data_set_size'] if 'data_set_size' in params else 1000
    data_set = create_data_set(env, data_set_size)
    #test_set = create_data_set(env, 5000)
    print("Completed creating data set.")
    #print("data set was created with n_squares_max: ", env.n_squares_max)
    
    
    #################################
    ###### TRAINING LOOP
    ###########################
    
    start_time = time.time()
    print("--- %s seconds ---" % (round(time.time() - start_time,2) ))
    print('Run for', max_epoch, 'iterations')
    
    for epoch in range(0, max_epoch):
          episode = model.episode
          model.episode = model.episode + 1
          if(epoch<=test_rate_changes_after+1):
            test_every_n = test_every_n_initial
          else:
            test_every_n = test_every_after          
          
          state_network = None          
          env.reset()
          loss = 0
          
          # Draw batch of espisodes from prepaired dataset
          env_img_list, env_action_list, ind_list, episode_length_list, env_task_list = get_batch_from_data_set(data_set, batch_size)
          #if(epoch==0):
          #  print(env_task_list)


          state_network_vis = None
          state_network_lang = None
          input_lang = torch.zeros( (batch_size, env.n_words+1) ).view(batch_size, env.n_words+1)

          if(epoch==0):
            print("epepisode_length_list: ", episode_length_list)
            for batch_ex in range(1):
              batch_ex_vector_for_each_timestep = [env_task_list[t][batch_ex].numpy().astype(int) for t in range(episode_length_list[batch_ex])]
              print("Task vector for batch-example: ", batch_ex)
              print(env.task_node_names)
              [print(tasky) for tasky in batch_ex_vector_for_each_timestep]

              batch_ex_actions_for_each_timestep = [env_action_list[t][batch_ex].numpy().astype(int) for t in range(episode_length_list[batch_ex])]
              print("env_action_list for batch-example: ", batch_ex)
              print(env.a_strings)
              [print(env.get_a_string_from_triple_onehot(actiony)) for actiony in batch_ex_actions_for_each_timestep]
          
          ##########################
          ###### EPISODE LOOP
          retain_graph = True
          for t in range(0, len(env_img_list)):
             
              state_network_vis, output_action, state_network_lang, output_lang = model(env_img_list[t],state_network_vis, input_lang, state_network_lang, env_task_list[t]) 
             
              Q_values = torch.cat((output_action, output_lang),1)
              
              loss += loss_fn(Q_values, env_action_list[t] )

              input_lang = copy(output_lang)
              input_lang = env_action_list[t][:,-env.n_words-1:]
              
              state_network_vis[0] = state_network_vis[0][ind_list[t]] #.detach()   #if you want to retain graph/LSTM-cell
              state_network_vis[1] = state_network_vis[1][ind_list[t]] #.detach()
              
              state_network_lang[0] = state_network_lang[0][ind_list[t]] #.detach()
              state_network_lang[1] = state_network_lang[1][ind_list[t]] #.detach()
              
              input_lang = input_lang[ind_list[t]]
              #model_lang_mem_forget_states_0 = model.lang_mem_forget_states[0].transpose(0,1)[ind_list[t]].transpose(0,1)
              #model_lang_mem_states_0 = model.lang_mem_states[0].transpose(0,1)[ind_list[t]].transpose(0,1)
              #model_lang_mem_forget_states_1 = model.lang_mem_forget_states[1].transpose(0,1)[ind_list[t]].transpose(0,1)
              #model_lang_mem_states_1 = model.lang_mem_states[1].transpose(0,1)[ind_list[t]].transpose(0,1)              
              #model.lang_mem_forget_states = (model_lang_mem_forget_states_0, model_lang_mem_forget_states_1 )
              #model.lang_mem_states = (model_lang_mem_states_0, model_lang_mem_states_1 )
              #env_task_list = env_task_list[ind_list[t]]
               
              # if(t%30==9):
              #   model.zero_grad()
              #   loss.backward(retain_graph=retain_graph)
              #   #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
              #   optimizer.step()
              #   average_loss += loss.data.numpy()
              #   #model_before_update = copy(model)
              #   state_network_vis[0] = state_network_vis[0].detach()
              #   state_network_vis[1] = state_network_vis[1].detach()
              #   state_network_lang[0] = state_network_lang[0].detach()
              #   state_network_lang[1] = state_network_lang[1].detach()
              #   input_lang.detach()
              #   env_task_list.detach()
              #   loss = 0   # this is important when retain_graph is on
              #   Q_values.detach()

              #   retain_graph = False

                

          average_loss += loss.data.numpy()
          #model_before_update = copy(model)
          model.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
          optimizer.step()          
          
          if((epoch+1)%1000==0):

            file_name = str(model.episode)  
            PATH = directory_path_models + file_name
            torch.save(model.state_dict(), PATH)
            
          early_criteria = (epoch+1) < 201 and (epoch+1)%20==0  
          if((epoch+1)%test_every_n==0):
              print("Episode: ", epoch+1)
              print("Learning rate: ", get_lr(optimizer))
              print("Average Loss in past ", test_every_n, " runs: ", average_loss/test_every_n)
              print("Update-rate: ", update_rate/test_every_n)              
              print("--- ", (str(datetime.timedelta(seconds=round(time.time() - start_time,2) )))  , " seconds ---"  )
              
              masters_all = test_env_on_every_task_and_N(env, model, episode, run_n, average_loss, test_every_n)
              model.train()
              
              print("---------------------------------------")
              print("--- ", (round(time.time() - start_time,2) ) ," seconds ---" )

                            
              ## Modify this if you want to have an changing learning rate: dependent on current learning rate or depending on the current loss
              if(average_loss/test_every_n < 0.15):
                  model.lr = 0.003 #model.lr #0.001 #0.05
                  #pass
              for g in optimizer.param_groups:
                  g['lr'] = model.lr
                  #pass              
              average_loss = 0

          if((epoch+1)%test_every_n==0 and masters_all):
              break
              

def test_env_on_every_task_and_N(env, model, episode, run_n, average_loss, test_every_n):              
      for task in env.task_list:
          print(" ")
          model.n_solved_for_m = {}

          task_n = env.task_list.index(task)
          #n_squares_max = env.n_squares_max_list[task_n]
          masters_all = True

          for n_ in range(1, env.test_for_n_squares_max[task_n]+1):
            model.n_solved_for_m[n_] = []
            env.task = task
            env.rand_n_squares = False
            env.sample_task = False
            env.n_squares_wished = n_
            env.reset()

            n_test_runs = 10                   

            if(env.task == "count_on"):
              n_test_runs = 1
              for addy in range(1,env.n_squares_max-n_+1):
                env.add_n = addy
                success_number, n_one_to_one_correspondences, n_right_number_order, variabilities = test_model(env, model, n_test_runs)
                model.result_tensory.add_episode_result(task, n_, episode, success_number/n_test_runs, run_n,n_one_to_one=n_one_to_one_correspondences, n_right_number_words=n_right_number_order,variabilities=variabilities)
                model.result_tensory.add_loss(average_loss/test_every_n, episode)
                if(success_number != n_test_runs):
                    masters_all = False
                print("")
            else:
              success_number, n_one_to_one_correspondences, n_right_number_order, variabilities = test_model(env, model, n_test_runs)
              model.result_tensory.add_episode_result(task, n_, episode, success_number/n_test_runs, run_n, n_one_to_one=n_one_to_one_correspondences, n_right_number_words=n_right_number_order, variabilities=variabilities)
              model.result_tensory.add_loss((average_loss/test_every_n), episode)
              if (success_number != n_test_runs and n_<=env.n_squares_max_list[task_n]):
                  masters_all = False

            
                   
          
      env.rand_n_squares = True
      env.sample_task = True
      env.n_squares_wished = -1

      return masters_all
              

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

      
def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)
  
def own_cross_entropy(inputy, target):
  logsoftmax = nn.LogSoftmax()
  return torch.mean(torch.sum(-target * logsoftmax(inputy), dim=1))
  

   

             