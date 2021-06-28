

from manage_results.run_schedules import *
import cProfile

#source_path = "/content/drive/MyDrive/embodied_counting/src/"
#directory_path = source_path + "manage_results/"

#file_name = "run_schedules.py"
#####################################
## AVERAGE AND SAVE MULTIPLE SCHEDULES FROM PRETRAINED
######################################

# Sequential Run 1: one run defines which tasks are going to be trained in parallel. This setup will then be run for n_replication times. We can run different setups in the run_list.
env_params = {
    'task_list': ["give_n_extended"],
    'distributed_squares': False,
    'max_n_squares': [9],
    'IsDecimal': True,
    'base': 10,
    'leading_zero': True,
    'img_size': 3
}

train_params = {
    'learning_rate': 1e-2,
    'batch_size': 16,
    'num_epochs': 15000,
    'data_set_size': 400
}

params = dict(env_params)
params.update(train_params)

runny = run(params)
#runny = run(["give_n_extended"], initial_lr=0.005, n_squares=[16], num_epochs = 10000)
run_list = []
run_list.append(runny)

## Curriculum
run_list = []
num_list = [9, 20, 30, 40]
for n in num_list:
    params['max_n_squares'] = n
    runny = run(params)
    run_list.append(runny)

#master_path = "/content/drive/My Drive/Embodied_counting/Results/count_all_events__give_n__recite_n__successor__more__1_TIMES__6252/multiple_tasks_1_to_9_21-02-19-14-02model-4289_/multiple_tasks_1_to_9_21-02-19-14-02model-4289_"

run_dfs, model = average_multiple_schedules(n_replications=1, run_list=run_list) #cProfile.run('average_multiple_schedules(n_replications=1, run_list=run_list)', sort='tottime')