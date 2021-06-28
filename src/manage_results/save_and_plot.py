######################################################
## PLOT ACCURACIES FOR MULTIPLE/ALL 1) TASKS AND 2) N ENTITIES
#####################################################

print("Load result managing..!")
import numpy as np
import os
import pandas as pd
import datetime
import torch
import matplotlib.pyplot as plt
import random
from environment.counting_tasks import *
from matplotlib.lines import Line2D

def isNaN(num):
    return num != num

def plot_accuracies_multiple_tasks_one_plot(df, fig, ax, xlim_manual = False, xlim=[0,200]):
    color_list = ["b", "g", "r", "c", "m", "y", "k", "tab:purple", "tab:orange", "tab:brown"]
    styl_list=[(0, ()), (0, (5, 5)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
    task_list_ = pd.unique(df.task.values)
    c_id = 0
    t_n = 0

    #fig, ax = plt.subplots(figsize=(16,6) )
    #fig=plt.figure(figsize=(16,6))
    #fig.show()
    #ax=fig.add_subplot(111)
    #titly = df.task[0]
    #ax.set_title(readable_task[df.task[0]], fontsize=font_size_title )

    if(xlim_manual == False): 
      xlim=[-10,df.episode.max()]

    font_size_title = 20
    font_size_legend = 15
    font_size_axis_label = 20
    font_size_axis_ticks = 20

    fm_list = dict.fromkeys(pd.unique(df.episode.values), 0)

    first_task = task_list_[0]
    second_task = task_list_[1]
    switch_task = df[df.task==first_task].episode.max()
    switch_task2 = df[df.task==second_task].episode.max()
    ax.axvline(x=switch_task, color = 'black',linewidth=0.6, alpha=0.6)

    middle_first_task =  ( int(switch_task/2), 1.3) 
    middle_second_task =  ( switch_task + int( (switch_task2-switch_task)/2 ), 1.3)

    beginning_first_task =  ( 0 + 100, 1.3) 
    beginning_second_task =  ( switch_task + 100, 1.3)

    #ax.annotate(readable_task[task_list_[0]], middle_first_task, fontsize=15, ha='center') #,xytext=(first_master,1.00 + fm_list[first_master]*0.07 )
    #ax.annotate(readable_task[task_list_[1]], middle_second_task, fontsize=15, ha='center')

    ax.annotate(readable_task[task_list_[0]], beginning_first_task, fontsize=15) #,xytext=(first_master,1.00 + fm_list[first_master]*0.07 )
    ax.annotate(readable_task[task_list_[1]], beginning_second_task, fontsize=15)

    l_id = 0
    for t in task_list_:

        
        c_id = 0
        for k in range(1,df.n_obj.values.max() + 1):

          if(t_n==0):
              labelly = "n= " + str(k)
              ax.plot(df[df.n_obj==k][df.task==t][df.episode<100000].episode.values, df[df.n_obj==k][df.task==t][df.episode<100000].accuracy.values, label=labelly, color = color_list[c_id] )
          else:
              ax.plot(df[df.n_obj==k][df.task==t][df.episode<100000].episode.values, df[df.n_obj==k][df.task==t][df.episode<100000].accuracy.values, color = color_list[c_id] )
    
          ax.set_xlim(xlim)
          ax.set_ylim([-0.1,1.4])
          ax.legend(prop={'size': font_size_legend}, loc='lower left')
          
          ax.xaxis.set_tick_params(labelsize=font_size_axis_ticks)
          ax.yaxis.set_tick_params(labelsize=font_size_axis_ticks)
          ax.set_yticks(np.arange(0,1.2,0.2))
          ax.set_xlabel("Episode",fontsize=font_size_axis_label)
          ax.set_ylabel("Accuracy",fontsize=font_size_axis_label)

          first_master = df[df.n_obj==k][df.task==t][df.accuracy==1].episode.min()
          ax.scatter(first_master, 1.0, color = color_list[c_id])

          if(not isNaN(first_master)):
              fm_list[first_master] += 1
              ax.annotate(k, (first_master, 1.0), fontsize=15,xytext=(first_master,1.00 + fm_list[first_master]*0.07 ), ha='center')
          c_id += 1
        t_n += 1  
      
    
               
    return fig




def plot_accuracies_multiple_tasks(df, fig, ax, xlim_manual = False, xlim=[0,200]):
    color_list = ["b", "g", "r", "c", "m", "y", "k", "tab:purple", "tab:orange", "tab:brown"]
    styl_list=[(0, ()), (0, (5, 5)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
    task_list_ = pd.unique(df.task.values)
    c_id = 0
    t_n = 0

    #fig, ax = plt.subplots(len(task_list_), 1, figsize=(16,12) , sharex=True)

    if(xlim_manual == False): 
      xlim=[-10,df.episode.max()]


    font_size_title = 20
    font_size_legend = 15
    font_size_axis_label = 20
    font_size_axis_ticks = 20

    for t in task_list_:

        l_id = 0
        for k in range(1,df.n_obj.values.max() + 1):
          labelly = "n= " + str(k)
          ax[t_n] = df[df.n_obj==k][df.task==t].plot(x="episode", y="accuracy", label=labelly, ax = ax[t_n], color = color_list[c_id], ls = styl_list[l_id] )

          ax[t_n].set_xlim(xlim)
          ax[t_n].set_ylim([-0.1,1.1])

          ax[t_n].legend(prop={'size': font_size_legend})
          ax[t_n].set_title(readable_task[t], fontsize=font_size_title )
          ax[t_n].xaxis.set_tick_params(labelsize=font_size_axis_ticks)
          ax[t_n].yaxis.set_tick_params(labelsize=font_size_axis_ticks)
          ax[t_n].set_xlabel("Episode",fontsize=font_size_axis_label)
          ax[t_n].set_ylabel("Accuracy",fontsize=font_size_axis_label)

          #ax[t_n].legend(prop={'size': 12})
          #ax[t_n].set_xlabel("Episode",fontsize=12)
          #ax[t_n].set_ylabel("Accuracy",fontsize=12)

          if(not isNaN(first_master)):
              first_master = df[df.n_obj==k][df.task==t][df.accuracy==1].episode.min()
              #ax[t_n].axvline(x=first_master, color = color_list[c_id], ls = styl_list[l_id],linewidth=0.6, alpha=0.6)
              ax[t_n].scatter(first_master, 1.0, color = color_list[c_id])

          l_id += 1
        t_n += 1  
        c_id += 1
    
    
        
    return fig

#ax = df[df.n_obj==k][df.task=='count_all_events'].plot(x="episode", y="losses", label='Total loss', ax = ax )  
#figy = plot_accuracies_multiple_tasks(df)  
#PATH_img_0,PATH_img_1  = create_path(df)

#PATH_img = PATH_img_0 + PATH_img_1 + '.png'
#figy.savefig(PATH_img, dpi=500)



######################################################
## PLOT ACCURACIES FOR SINGLE TASKS AND 2) N ENTITIES
#####################################################

def plot_accuracies_single_task(df, fig, ax, xmin=0):
      #color_list = ["b", "g", "r", "c", "m", "y", "k", "tab:purple", "tab:orange", "tab:brown"]
      color_list = []

      for i in range(df.n_obj.values.max()):
          color_list.append('#%06X' % random.randint(0, 0xFFFFFF))

      styl_list=[(0, ()), (0, (5, 5)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
      task_list_ = pd.unique(df.task.values)
      c_id = 0
      t_n = 0
 
      xlim=[xmin-10,df.episode.max()]

      #fig, ax = plt.subplots(figsize=(16,6) )
      font_size_title = 20
      font_size_legend = 15
      font_size_axis_label = 20
      font_size_axis_ticks = 20

      

      fm_list = dict.fromkeys(pd.unique(df.episode.values), 0)

      if(xmin==0):
          ax.set_title(readable_task[df.iloc[0].task], fontsize=font_size_title )
      else:
          beginning_first_task =  ( xmin + 100, 1.3) 
          ax.annotate(readable_task[task_list_[0]], beginning_first_task, fontsize=15)


      #t = "touch_all_objects"
      #df.losses = df.losses.astype(float)

      l_id = 0
      plot_dense_until = 8
      plot_every_k_then = 4
      ks_to_plot = [k_i for k_i in range(1,min(df.n_obj.values.max() + 1,plot_dense_until))]
      if(df.n_obj.values.max()>plot_dense_until):
        k_i = plot_dense_until + plot_every_k_then
        while(k_i < df.n_obj.values.max()):
          ks_to_plot.append(k_i)
          k_i += plot_every_k_then
        ks_to_plot.append(df.n_obj.values.max())
      
      for k in ks_to_plot:
        labelly = "n= " + str(k)
        ax = df[df.n_obj==k].plot(x="episode", y="accuracy", label=labelly, ax = ax, color = color_list[c_id])
        #ax = df[df.n_obj==k][df.episode>0].plot(x="episode", y="losses", label="Loss", ax = ax)
        #ax = df[df.episode<100000].losses.values.plot(ax=ax)
        #ax = df[df.episode<100000].losses.plot(ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim([-0.1,1.4])

        if(xmin==0):
            ax.legend(prop={'size': font_size_legend})
        else:
            ax.legend(prop={'size': font_size_legend}, loc='lower left')
        
        ax.xaxis.set_tick_params(labelsize=font_size_axis_ticks)
        ax.yaxis.set_tick_params(labelsize=font_size_axis_ticks)
        ax.set_yticks(np.arange(0,1.2,0.2))
        ax.set_xlabel("Episode",fontsize=font_size_axis_label)
        ax.set_ylabel("Accuracy",fontsize=font_size_axis_label)

        first_master = df[(df.n_obj==k) & (df.accuracy==1)].episode.min()
        #first_master = df[df.n_obj==k][df.accuracy==1].episode.min()
        #df.loc[(df["B"] > 50) & (df["C"] == 900), "A"]
        #ax[t_n].axvline(x=first_master, color = color_list[c_id], ls = styl_list[l_id],linewidth=0.6, alpha=0.6)
        ax.scatter(first_master, 1.0, color = color_list[c_id],)

        if(not isNaN(first_master)):
            fm_list[first_master] += 1
            ax.annotate(k, (first_master, 1.0), fontsize=15,xytext=(first_master,1.00 + fm_list[first_master]*0.07 ), ha='center')
        c_id += 1

        l_id += 1
      t_n += 1 

      

      return fig

    
    
def plot_sub_tasks_single_task(df, fig, ax, xmin=0):
      color_list = ["b", "g", "r", "c", "m", "y", "k", "tab:purple", "tab:orange", "tab:brown"]
      styl_list=[(0, ()), (0, (5, 5)),(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1))]
      task_list_ = pd.unique(df.task.values)
      c_id = 0
      t_n = 0
 
      xlim=[xmin-10,df.episode.max()]

      #fig, ax = plt.subplots(figsize=(16,6) )
      font_size_title = 20
      font_size_legend = 15
      font_size_axis_label = 20
      font_size_axis_ticks = 20

      

      fm_list = dict.fromkeys(pd.unique(df.episode.values), 0)

      if(xmin==0):
          ax.set_title(readable_task[df.iloc[0].task], fontsize=font_size_title )
      else:
          beginning_first_task =  ( xmin + 100, 1.3) 
          ax.annotate(readable_task[task_list_[0]], beginning_first_task, fontsize=15)


      #t = "touch_all_objects"
      #df.losses = df.losses.astype(float)

      l_id = 0
      for k in range(1,df.n_obj.values.max() + 1):
        labelly_one_to_one = "One-to-one correspondence" 
        labelly_numbers = "Correct number sequence" 
        labelly_variabilities = "Action variability"

        ax = df[df.n_obj==k].plot(x="episode", y="n_one_to_ones", ax = ax,label=labelly_one_to_one,  color = 'blue')
        ax = df[df.n_obj==k].plot(x="episode", y="n_right_number_words", ax = ax,label=labelly_numbers, color = 'red')
        ax = df[df.n_obj==k].plot(x="episode", y="variabilities", ax = ax,label=labelly_numbers, color = 'green')        

        #ax = df[df.n_obj==k][df.episode>0].plot(x="episode", y="losses", label="Loss", ax = ax)
        #ax = df[df.episode<100000].losses.values.plot(ax=ax)
        #ax = df[df.episode<100000].losses.plot(ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim([-0.1,1.4])

        if(xmin==0):
            ax.legend(prop={'size': font_size_legend}, loc='lower right')
        else:
            ax.legend(prop={'size': font_size_legend}, loc='lower right')
        
        ax.xaxis.set_tick_params(labelsize=font_size_axis_ticks)
        ax.yaxis.set_tick_params(labelsize=font_size_axis_ticks)
        ax.set_yticks(np.arange(0,1.2,0.2))
        ax.set_xlabel("Episode",fontsize=font_size_axis_label)
        ax.set_ylabel("Accuracy",fontsize=font_size_axis_label)


        
        ax.get_legend().remove()
        
        
        legend_elements = [Line2D([0], [0], color='blue', lw=2, label=labelly_one_to_one),Line2D([0], [0], color='red', lw=2, label=labelly_numbers),Line2D([0], [0], color='green', lw=2, label=labelly_variabilities)]

        ax.legend(handles=legend_elements, prop={'size': font_size_legend}, loc='lower right')
        #ax.plot([], [], label=labelly_one_to_one, color='blue')
        #ax.plot([], [], label=labelly_numbers, color='red')
        
        
        #ax[t_n].axvline(x=first_master, color = color_list[c_id], ls = styl_list[l_id],linewidth=0.6, alpha=0.6)
        

        '''
        first_master = df[df.n_obj==k][df.n_one_to_ones==1].episode.min()
        ax.scatter(first_master, 1.0, color = 'blue')
        if(not isNaN(first_master)):
            fm_list[first_master] += 1
            ax.annotate(k, (first_master, 1.0), fontsize=15,xytext=(first_master,1.00 + fm_list[first_master]*0.07 ), ha='center')
            
        first_master = df[df.n_obj==k][df.n_right_number_words==1].episode.min()
        ax.scatter(first_master, 1.0, color = 'red')
        if(not isNaN(first_master)):
            fm_list[first_master] += 1
            ax.annotate(k, (first_master, 1.0), fontsize=15,xytext=(first_master,1.00 + fm_list[first_master]*0.07 ), ha='center')            
        '''
        c_id += 1

        l_id += 1
      t_n += 1 

      

      return fig    
    
    
    
    
    
    
    
    
    
#figy = plot_accuracies_single_task(df)
#ax = df[df.n_obj==k][df.task=='count_all_events'].plot(x="episode", y="losses", label='Total loss', ax = ax )  


######################
## Plot accuracies of current df
###############################


def plot_transfer_effect(df_continual, df_single):

    fig, ax = plt.subplots(2, 1, figsize=(16,12) , sharex=False)
    
    plot_accuracies_multiple_tasks_one_plot(df_continual, fig=fig, ax=ax[0])
    xmin = -df_continual[df_continual.runs==0].episode.max()
    plot_accuracies_single_task(df_single, fig=fig, ax=ax[1], xmin=xmin)

    return fig

def plot_accuracies(df,xmin=0):
    task_list_ = pd.unique(df.task.values)
    task_list_length = len(task_list_)
    fig_2, ax_2 = plt.subplots(figsize=(16,6) )

    if(task_list_length == 1): 
        fig, ax = plt.subplots(figsize=(16,6) )
        #fig_2, ax_2 = plt.subplots(figsize=(16,6) )
        fig = plot_accuracies_single_task(df, fig=fig, ax=ax, xmin=xmin)
        fig_2 = plot_sub_tasks_single_task(df, fig=fig_2, ax=ax_2, xmin=xmin)
    else:
        if(df.runs.max()>1):
            fig, ax = plt.subplots(len(task_list_), 1, figsize=(16,12) , sharex=True)
            plot_accuracies_multiple_tasks(df, fig=fig, ax=ax)
            print("print multiple tasks")
        else:
            fig, ax = plt.subplots(figsize=(16,6) )
            #ax=fig.add_subplot(111)
            plot_accuracies_multiple_tasks_one_plot(df, fig=fig, ax=ax)
        #plot_accuracies_multiple_tasks(df)
        #plot_accuracies_single_task(df)

    plt.close()
    
    return fig, fig_2




def plot_confusion_matrix(model):
    fig, ax = plt.subplots(figsize=(16,6) )
    data_dict = {key: [val_i for val_i in val if val_i != -1] for key, val in model.n_solved_for_m.items()}
    data = [values for key, values in data_dict.items()]
    #data_array = np.array(data)
    data_mean = [np.mean(data[i]) for i in range(len(data))]
    keys = [key for key, values in data_dict.items()]

    font_size_title = 20
    font_size_legend = 15
    font_size_axis_label = 20
    font_size_axis_ticks = 20

    ax.boxplot(data, whis=[0, 100])
    max_n = len(data)
    one_to_n_range = np.arange(1, max_n + 1)
    ax.scatter(one_to_n_range, data_mean, color='black')

    ax.plot(one_to_n_range, one_to_n_range, color='black')
    ax.set_xlabel("Demanded Number of Objects",fontsize=font_size_axis_label)
    ax.set_ylabel("Given Number of Objects",fontsize=font_size_axis_label)
    ax.yaxis.set_tick_params(labelsize=font_size_axis_ticks)
    ax.xaxis.set_tick_params(labelsize=font_size_axis_ticks)
    ax.set_yticks(one_to_n_range)


    return fig





def save_and_plot(df, model=None, run_time=None, extra_folder=""):
    #drive.mount('/content/drive')

    #Create Basic Path
    directory_path, file_name = create_path(df, model, extra_folder)

    #Save df
    PATH = directory_path + file_name + ".pkl"
    df.to_pickle(PATH)

    if(model is not None):
        #Save model
        PATH = directory_path + file_name
        print("Model-Path = ", PATH)
        model.model_path = PATH
        torch.save(model.state_dict(), PATH)

    if(run_time is not None):
        #Save run time
        PATH = directory_path + str(sec_to_hours(run_time) )
        fily = open(PATH,"w")
        fily.close()

    #Plot accuracies
    fig, fig2 = plot_accuracies(df)

    #Save images    
    PATH = directory_path + file_name + ".png"
    fig.savefig(PATH, dpi=400)
    
    PATH = directory_path + "sub_task_stages.png"
    fig2.savefig(PATH, dpi=400)

    #Plot Confusion matrices
    fig3 = plot_confusion_matrix(model)
    PATH = directory_path + "confusion_plot.png"
    fig3.savefig(PATH, dpi=400)



def plot_and_save_transfer_effect(df_continual, df_single):
    #Plot accuracies
    plt.close('all')
    fig = plot_transfer_effect(df_continual, df_single)

    #Save images    
    #Create Basic Path
    #extra_folder = create_extra_folder(run_list, n_replications)
    extra_folder = "TRANSFER_"
    extra_folder += create_extra_folder_for_transfer(task_1[0], task_2[0], n_replications, create=False)
    #directory_path, file_name = create_path(df, extra_folder=extra_folder)
    directory_path = "../Results/" + extra_folder
    os.mkdir(directory_path)
    file_name = extra_folder[:-1]
    #Save
    #drive.mount('/content/drive')
    PATH = directory_path + file_name + ".png"
    fig.savefig(PATH, dpi=400)
    #fig.save(PATH)

def save_schedule_specifications(run_list, model, extra_folder):

    directory_path = "../Results/" + extra_folder
    #os.mkdir(directory_path)
    PATH = directory_path + "schedule_specifications.txt"
    print("schedule spec PATH: ", PATH)

    fily = open(PATH,"w")
    # Run specifications
    run_list_text = "Specifications about the running schedule \n \n"
    for r in range(len(run_list)):
        run_list_text += "Run " + str(r) + "\n"
        run_list_text += "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in run_list[r].params.items()) + "}" + "\n \n"
    print(run_list_text)
    #f.write(run_list_text)

    # Model specifications
    model_string = "Specifications about the Neural Network Architecture \n \n" + str(model.model_id)

    #Whole
    whole_text = run_list_text + model_string
    fily.write(whole_text)
    fily.close()

#save_and_plot(df)

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d

def average_dfs(run_dfs):
    summed_df = run_dfs[0].copy()
    total_accuracy = run_dfs[0]['accuracy']
    total_loss = run_dfs[0]['losses']

    for r in range(1,len(run_dfs)):
      total_accuracy += run_dfs[r]['accuracy']
      total_loss += run_dfs[r]['losses']

    summed_df['accuracy'] = total_accuracy / len(run_dfs)
    summed_df['losses'] = total_loss / len(run_dfs)

    return summed_df


##############################
## CREATE PATH - BASIS
##############################




def create_extra_folder(run_list, n_runs,from_pretrained=False):
    #drive.mount('/content/drive')
    folder_id = np.random.randint(1,10000)
    extra_folder = ""
    for r in run_list:
        for t in r.task_list:
            extra_folder += t + "__"
        extra_folder += "THEN_"
    extra_folder = extra_folder[:-5]
    if(from_pretrained):
      	extra_folder += "_from_pretrained_"
    extra_folder += str(n_runs) + "_TIMES__" + str(folder_id) + "/"
    if(len(extra_folder)>30):
      extra_folder = extra_folder[:30] + str(folder_id) + "/"
    directory_path = "../Results/" + extra_folder
    os.mkdir(directory_path)
    
    #Create Extra-folder for gifs and action-list:
    directory_path += "GIFs_and_ACTIONS/"
    os.mkdir(directory_path)

    return extra_folder

def create_extra_folder_for_transfer(task1, task2, n_replications, create=True):
    #drive.mount('/content/drive')
    folder_id = np.random.randint(1,10000)
    extra_folder = ""
    extra_folder += str(task1)
    extra_folder += "_THEN_"
    extra_folder += str(task2)

    extra_folder += "__" + str(n_replications) + "_TIMES__" + str(folder_id) + "/"
    directory_path = "../Results/" + extra_folder
    if(create==True):
        os.mkdir(directory_path)

    return extra_folder

def create_path(df, model=None, extra_folder = ""):
    #drive.mount('/content/drive')
    # Get task string
    task_string = ""
    if(len(pd.unique(df.task.values))>1 ):
      task_string = "multiple_tasks"
    else:
      task_string = pd.unique(df.task.values)[0] 

    # Get n string
    min_n = df.n_obj.values.min()
    max_n = df.n_obj.values.max()
    n_string = "_" + str(min_n)+"_to_"+str(max_n) + "_"

    # Get date
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")



    # Create full path
    directory_path = "../Results/" + extra_folder
    #os.mkdir(directory_path)
    file_name = task_string + n_string + date 

    if(model is not None):
        # Get model-id
        model_id = "model-" + str(model.model_id) + "_"
        file_name += model_id 
    
    directory_path = directory_path + file_name + "/"
    os.mkdir(directory_path)
    
    return directory_path, file_name
  
#directory_path, file_name = create_path(df)


