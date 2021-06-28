######################################################
### DEFINE: RUN --> SCHEDULE (multiple sequential runs) --> MULTIPLE SAME SCHEDULES (AVERAGE)
#######################################################################

from environment.counting_tasks import *
from environment.minimal_count_environment import *
from imports import *
from models.ConvAtt_LangConvLSTM_with_graph import *
from manage_results.save_and_plot import *
from  train_and_test.train_model_original import *
CUDA_bool = False


print("Load run, schedules ..! ")

class run():
    def __init__(self, params):
        self.params = params
        self.task_list = params['task_list']
        self.initial_lr = params['learning_rate']
        self.n_squares = params['max_n_squares']
        self.num_epochs = params['num_epochs']
        self.batch_size = params['batch_size']
        self.IsDecimal = params['IsDecimal']
        self.base = params['base']
        self.leading_zero = params['leading_zero']
        #self.run_model = run_model

    def to_text(self):
        # First convert to dict, then easily to text
        dicty = {
                "Task list": self.task_list,
                "Maximum number of countable objects": self.n_squares,
                "Number of epochs": self.num_epochs,
                "Initial learning rate": self.initial_lr    
                } 

        return str(dicty)


    def to_dict(self):
        # First convert to dict, then easily to text
        dicty = {
                "Task list": self.task_list,
                "Maximum number of countable objects": self.n_squares,
                "Number of epochs": self.num_epochs,
                "Initial learning rate": self.initial_lr    
                } 

        return dicty

def run_training_schedule(run_list, model):
    #drive.mount('/content/drive')
    
    #Run sequentially on same model
    for n in range(len(run_list)):
        print("----------------------")
        print("---- SWITCH TASKS --------")
        print("-----------------------")
        model.lr = run_list[n].initial_lr
        train_model(task_list=run_list[n].task_list, n_squares_ = run_list[n].n_squares, num_epochs = run_list[n].num_epochs, run_n = n, model = model, params = run_list[n].params) #run_n = n
    df = model.result_tensory.create_panda_df()


    return df, model

def average_multiple_schedules(n_replications, run_list,initial_model_path=None):


    c = 2           # Input size
    d = 5           # Hidden size
    lr_dummy = 0.1
    env = CountEnv(task = run_list[0].task_list[0], n_squares = run_list[0].n_squares, display = "game", save_epoch = True, params=run_list[0].params )
    
    model = LangConvLSTMCell(c,d,env,lr_dummy)
    from_pretrained = False
    if(initial_model_path is not None):
    	model.load_state_dict(torch.load(initial_model_path))   #.cuda()
    	from_pretrained = True
    
    

    run_dfs = []
    extra_folder = create_extra_folder(run_list, n_replications, from_pretrained=from_pretrained)
    
    save_schedule_specifications(run_list, model, extra_folder)
    

    total_time_1 = time.time()

    for n in range(n_replications):
        run_number = "======== RUN NUMBER " + str(n) + "  ========"
        print("")
        print("========================")
        print(run_number)
        print("========================")
        if(CUDA_bool == True):
           model = LangConvLSTMCell(c,d,env,lr_dummy).cuda()   #.cuda()
        else: 
           model = LangConvLSTMCell(c,d,env,lr_dummy)
        if(initial_model_path is not None):
    	    model.load_state_dict(torch.load(initial_model_path)) 
        ind_time_1 = time.time()
        
        #directory_path, file_name = create_path(df, model, extra_folder)
        #model.model_path = "/content/drive/MyDrive/embodied_counting/Results/" + extra_folder + str(n) + "_"
        
        df, model = run_training_schedule(run_list, model)
        #print("Model trained and saved at: ", model.model_path)
        ind_time_2 = time.time()
        diff_time = ind_time_2 - ind_time_1
        run_dfs.append(df)        
        save_and_plot(df, model, run_time=diff_time, extra_folder=extra_folder)

    extra_folder += "AVERAGE_"
    # Save time
    total_time_2 = time.time()
    diff_time = total_time_2 - total_time_1
    avg_df = average_dfs(run_dfs)
    
    #plt.close('all')
    save_and_plot(avg_df,model, run_time=diff_time, extra_folder=extra_folder)
    return avg_df, model


def check_transfer_effects(task_1, task_2, task_1_episodes, task_2_episodes, task_1_lr, task_2_lr, n_replications, n_objects=9):
    # Sequential Run 1
    runny = run(task_1, initial_lr=task_1_lr, n_squares=n_objects, num_epochs = task_1_episodes)
    # Sequential Run 1
    runny2 = run(task_2, initial_lr=task_2_lr, n_squares=n_objects, num_epochs = task_2_episodes)
    run_list = []
    run_list.append(runny)
    run_list.append(runny2)
    df_continual = average_multiple_schedules(n_replications=n_replications, run_list=run_list)

    #Scratch run
    runny3 = run(task_2, initial_lr=task_2_lr, n_squares=n_objects, num_epochs = task_2_episodes)
    run_list = []
    run_list.append(runny3)
    df_single = average_multiple_schedules(n_replications=n_replications, run_list=run_list)

    return df_continual, df_single