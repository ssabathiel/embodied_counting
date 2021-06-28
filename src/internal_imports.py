import os 
# Import: imports
directory_path = "/content/drive/My\ Drive/Embodied_counting/src/"

file_name = "imports.py"
PATH = directory_path + file_name
#%run -i $PATH
#!python $PATH

#command = 'run ' + PATH
#os.system(command)
#import PATH



# Import: environment
directory_path = "/content/drive/My\ Drive/Embodied_counting/src/environment/"

file_name = "count_environment.py"
PATH = directory_path + file_name
#%run -i $PATH
#!python $PATH

command = 'python ' + PATH
os.system(command)


file_name = "solving_algorithms.py"
PATH = directory_path + file_name
#%run -i $PATH



# Import: models
directory_path = "/content/drive/My\ Drive/Embodied_counting/src/models/"

file_name = "LangConvLSTM.py"
PATH = directory_path + file_name
#%run -i $PATH


# Import: train_and_test
directory_path = "/content/drive/My\ Drive/Embodied_counting/src/train_and_test/"

file_name = "train_model.py"
PATH = directory_path + file_name
#%run -i $PATH

file_name = "test_model.py"
PATH = directory_path + file_name
#%run -i $PATH

file_name = "env_to_pytorch_interface.py"
PATH = directory_path + file_name
#%run -i $PATH


# Import: manage_results
directory_path = "/content/drive/My\ Drive/Embodied_counting/src/manage_results/"

file_name = "run_schedules.py"
PATH = directory_path + file_name
#%run -i $PATH

file_name = "save_and_plot.py"
PATH = directory_path + file_name
#%run -i $PATH


