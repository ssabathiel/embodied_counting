#################################
## Try weight-node matrix
#################################

print("Load Hinton-diagrams...!!")
import numpy as np


##
#Example one run-through

##############
## Get graphs for different tasks
###############################
#PATH = "/content/drive/My Drive/Embodied_counting/Results/count_all_objects__count_all_events__give_n__recite_n__1_TIMES__1038/multiple_tasks_1_to_9_20-01-26-08-43model-1002_/multiple_tasks_1_to_9_20-01-26-08-43model-1002_"
#PATH = "C:/Users/silvests/Embodied_counting/Results/count_all_objects__1_TIMES__7989/count_all_objects_1_to_9_20-03-13-12-51model-6645_/count_all_objects_1_to_9_20-03-13-12-51model-6645_"
# PATH = "C:/Users/silvests/Embodied_counting/Results/master_all/master_all/multiple_tasks_1_to_9_20-03-31-01-29model-8138_"
# n_squares = 4


# graph_recite_n = get_graph_from_task_trial(PATH, "recite_n", n_squares)
# graph_count_events = get_graph_from_task_trial(PATH, "count_all_events", n_squares)
# graph_count_objects = get_graph_from_task_trial(PATH, "count_all_objects", n_squares)
# graph_give_n = get_graph_from_task_trial(PATH, "give_n", n_squares)

# graph_list_recite_n = GraphListClass(graph_recite_n, "Recite-N")
# graph_list_count_all_events = GraphListClass(graph_count_events, "Count all events")
# graph_list_count_all_objects = GraphListClass(graph_count_objects, "Count all objects")
# graph_list_give_n = GraphListClass(graph_give_n, "Give-N")

# multiple_graph_lists = [graph_list_count_all_events, graph_list_count_all_objects, graph_list_give_n]


# ############################
# ## Plot Hinton diagrams
# ############################
# asked_numbers=[1,2,3,4,'no_count', 'no_count', 'no_count','no_count', 'no_count']
# fig = hinton_from_multiple_graph_lists(multiple_graph_lists, asked_numbers=asked_numbers, layers=(1,2),axis=None, figy=None, weight_encoding='size',max_weight=None)

# #for n in range(1,4):
# asked_numbers=[1,2,3,4]
# #asked_numbers=[1]
# fig = hinton_from_multiple_graph_lists(multiple_graph_lists, 
#                                        asked_numbers=asked_numbers, 
#                                        layers=(0,2),
#                                        axis=None, figy=None, 
#                                        weight_encoding='size',
#                                        max_weight=None,
#                                       highlight_number=[25,27])



def hinton(matrix,axis=None, figy=None, input_nodes=None,input_node_description="",output_nodes=None,output_names=None, max_weight=None, ax=None, weight_encoding='size', graphy=None, input_node_names=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    #ax = ax if ax is not None else plt.gca()
    
    
    
    scale_fig=0.7
    fig_sizy=(matrix.shape[0]*scale_fig,matrix.shape[1]*scale_fig)
    fig, ax = plt.subplots(figsize=fig_sizy)
    
    if(figy is not None and axis is not None):
        fig = figy
        ax = axis
    
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray') #gray
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    ## Extend patch to left
    text_dist = 1
    text_rect = plt.Rectangle([-7, 0], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)

    
    if(output_nodes is not None):
        dist_to_weights = 3
        for y in range(output_nodes.size):
            max_ampl=2 ** np.ceil(np.log(np.abs(output_nodes).max()) / np.log(2))
            color, size = get_rect_size_and_color(amplitude=output_nodes[y],max_amplitude=max_ampl,amplitude_encoding=weight_encoding, fixed_size=1)
            rect = plt.Rectangle([-dist_to_weights - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            if(len(str(output_names[y]))>5):
                extra_distance = 1.4
            else:
                extra_distance = 0
            if(output_names is not None):
                ax.text(-dist_to_weights-len(str(output_names[y]))/4-1.2-extra_distance, y+0.3, str(output_names[y]), fontsize=12)
            
    if(input_nodes is not None):
        dist_to_weights = 3
        for x in range(input_nodes.size):
            max_ampl=2 ** np.ceil(np.log(np.abs(input_nodes).max()) / np.log(2))
            color, size = get_rect_size_and_color(amplitude=input_nodes[x],max_amplitude=max_ampl,amplitude_encoding=weight_encoding, fixed_size=1)
            rect = plt.Rectangle([x - size / 2, -dist_to_weights-size/2], size, size,
                             facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            if(input_node_names is not None):
              ax.text(x - size / 2, -dist_to_weights-size/2, str(input_node_names[x]),fontsize=12)

    for (x, y), w in np.ndenumerate(matrix):
        #print(x,y)
#         if(weight_encoding=='size'):
#             color = 'white' if w > 0 else 'black'
#             size = np.sqrt(np.abs(w) / max_weight)        
#         elif(weight_encoding=='transparency'):
#             transparency = (np.abs(w) / max_weight)*(np.abs(w) / max_weight)
#             if(w>0):
#                 color = 'rgba(1,1,1,' + str(transparency) + ')'
#                 #color = (1,1,1,transparency)
#                 color = (0,0,1,transparency)
#             else:
#                 color = 'rgba(0,0,0,' + str(transparency) + ')'
#                 color = (0,0,0,transparency)
#                 color = (1,0,0,transparency)
                
#             size = 1 #np.sqrt(np.abs(w) / max_weight)

        color, size = get_rect_size_and_color(amplitude=w,max_amplitude=max_weight,amplitude_encoding=weight_encoding, fixed_size=1)    
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        
    # Add "W" description
    text_dist = 1
    text_rect = plt.Rectangle([matrix.shape[0]/2-2, matrix.shape[1]+text_dist], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)
    ax.text(matrix.shape[0]/2-2, matrix.shape[1]+text_dist, "W", fontsize=18)
    
    # Add representation description
    text_dist = 1
    dist_to_weights = 1
    text_rect = plt.Rectangle([matrix.shape[0]/2-2, -text_dist-dist_to_weights-size-2], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)
    ax.text(matrix.shape[0]/2-7.0, -text_dist-dist_to_weights-size, input_node_description, fontsize=18)
        
    ax.autoscale_view()
    ax.invert_yaxis()
    #scale_fig=6
    #ax.fig_size=(x*scale_fig,y*scale_fig)
    
    return fig


def get_rect_size_and_color(amplitude,max_amplitude,amplitude_encoding='size', fixed_size=None):
        if(amplitude_encoding=='size'):
            color = 'white' if amplitude > 0 else 'black'
            size = np.sqrt(np.abs(amplitude) / max_amplitude)  #(np.abs(amplitude) / max_amplitude)        
        elif(amplitude_encoding=='transparency'):
            transparency_unclipped = (np.abs(amplitude) / max_amplitude) #(np.abs(amplitude) / max_amplitude)*(np.abs(amplitude) / max_amplitude)
            transparency = np.clip(transparency_unclipped, a_min = 0.0, a_max = 1.0) 
            if(np.isnan(transparency) or np.isinf(transparency)):
                transparency = 0.0
            if(amplitude>0):
                color = 'rgba(1,1,1,' + str(transparency) + ')'
                #color = (1,1,1,transparency)
                color = (0,0,1,transparency)
            else:
                color = 'rgba(0,0,0,' + str(transparency) + ')'
                color = (0,0,0,transparency)
                color = (1,0,0,transparency)
            size = fixed_size
        return color, size
    

    
def hinton_from_multiple_graph_lists(multiple_graph_lists, 
                                     asked_numbers, 
                                     layers=(0,1),
                                     axis=None, 
                                     figy=None, 
                                     weight_encoding='size',
                                     max_weight=None,
                                    highlight_number=None,
                                    highlight_output_number=None):
#(matrix,axis=None, figy=None, input_nodes=None,input_node_description="",output_nodes=None,output_names=None, max_weight=None, ax=None, weight_encoding='size', graphy=None):
    """Draw Hinton diagram for visualizing a weight matrix."""  
    
    
    small_font_size = 14
    
    graphy = multiple_graph_lists[0].graph_list[0]
    scale_fig=0.5
    matrix=graphy.weights[layers].transpose()
    n_tasks = len(multiple_graph_lists)
    n_asked_numbers = len(asked_numbers)
    
    time_step = multiple_graph_lists[0].time_steps_count_and_no_count[asked_numbers[0]][0]
    
    output_nodes = multiple_graph_lists[0].node_activations[layers[1]][time_step]
    input_node_description = multiple_graph_lists[0].graph_list[0].layer_descriptions[layers[0]][0]
    output_names = np.asarray([i for i in range(len(graphy.nodes_in_layer[layers[1]]))])
    
    #output_nodes = multiple_graph_lists[0].output_nodes_list[time_step]
    #input_node_description = multiple_graph_lists[0].input_node_description[0]
    #create self.graphs_of_asked_number[] in GraphListClass or same from def compare
    
    n_rows = len(asked_numbers)
    n_cols = len(multiple_graph_lists)
    

    fig_sizy=(matrix.shape[0]*scale_fig,matrix.shape[1]*scale_fig+n_asked_numbers*n_tasks)
    fig, ax = plt.subplots(figsize=fig_sizy)
    
    if(figy is not None and axis is not None):
        fig = figy
        ax = axis
    
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray') #gray
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    ## Extend patch to left
    text_dist = 1
    text_rect = plt.Rectangle([-14, 0], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)
    if(len(multiple_graph_lists)>1):
        distance_between_asked_numbers = 3.0
        extra_dist_2 = 0.0
    else:
        distance_between_asked_numbers = 0.0
        extra_dist_2 = 3.0
    n_no_counts=0
    
    
    if(matrix.shape[0]>matrix.shape[1]):
         #print("low matrix")
         extra_dist_3=0.0
    else:
         #print("hight matrix")
         extra_dist_3=1.5
        
    highlight_upper_corners = {}
    highlight_lower_corners = {}
    
        
    for a in range(len(asked_numbers)):
            
        for tasky in range(len(multiple_graph_lists)):
                if(asked_numbers[a]=='no_count'):
                    time_step = multiple_graph_lists[tasky].time_steps_count_and_no_count[asked_numbers[a]][n_no_counts]
                    if(tasky==len(multiple_graph_lists)-1):
                        n_no_counts+=1
                else:
                    time_step = multiple_graph_lists[tasky].time_steps_count_and_no_count[asked_numbers[a]][0]
                #input_nodes = multiple_graph_lists[tasky].input_nodes_list[time_step]
                input_nodes = multiple_graph_lists[tasky].node_activations[layers[0]][time_step]
                dist_to_weights = 3
                for x in range(input_nodes.size):
                    max_ampl=2 ** np.ceil(np.log(np.abs(input_nodes).max()) / np.log(2))
                    color, size = get_rect_size_and_color(amplitude=input_nodes[x],max_amplitude=max_ampl,amplitude_encoding=weight_encoding, fixed_size=1)
                    rect = plt.Rectangle([x - size / 2, -dist_to_weights-size/2-tasky-n_tasks*a-a*distance_between_asked_numbers], size, size,
                                     facecolor=color, edgecolor=color)
                    ax.add_patch(rect)
                    
                    ## remember last node position for higlighting if needed
                    if(highlight_number is not None):
                        if(x in highlight_number and a==len(asked_numbers)-1 and tasky==len(multiple_graph_lists)-1):
                            highlight_upper_corners[x] = [x-size/2, -dist_to_weights-size/2-tasky-n_tasks*a-a*distance_between_asked_numbers]
                if(len(multiple_graph_lists)>1):
                    task_spec = multiple_graph_lists[tasky].task + ":"
                    #ax.text(-dist_to_weights-len(task_spec)/2.4-1.1,-dist_to_weights-tasky, task_spec, fontsize=12)
                    ax.text(-dist_to_weights+1.7,-dist_to_weights-tasky-n_tasks*a+1/4-a*distance_between_asked_numbers, task_spec, fontsize=small_font_size,horizontalalignment='right')
        
                
        if(type(asked_numbers[a])==int):
            asked_number_description = "Entity Count " + str(asked_numbers[a]) + " :"
        if(asked_numbers[a]=='no_count'):
            asked_number_description = "No Entity"  + ":"
        #ax.text(-dist_to_weights-7.5-extra_dist_2,-dist_to_weights-tasky-n_tasks*a+1/4-(a+0.5)*distance_between_asked_numbers, asked_number_description, fontsize=small_font_size,color='blue',horizontalalignment='left')
        
        if(len(multiple_graph_lists)>1):
            # put distance between asked numbers
            rect = plt.Rectangle([x - size / 2, -dist_to_weights-size/2-tasky-n_tasks*a-distance_between_asked_numbers], size, size,
            facecolor='gray', edgecolor='gray')
            ax.add_patch(rect)


    # node-enumeration
    for x in range(input_nodes.size):
        ax.text(x, -dist_to_weights-size/2-tasky-n_tasks*a-a*distance_between_asked_numbers-1, str(x), fontsize=small_font_size,horizontalalignment='center')


    
    if(output_nodes is not None):
        dist_to_weights = 3
        for y in range(output_nodes.size):
            max_ampl=2 ** np.ceil(np.log(np.abs(output_nodes).max()) / np.log(2))
            color, size = get_rect_size_and_color(amplitude=output_nodes[y],max_amplitude=max_ampl,amplitude_encoding=weight_encoding, fixed_size=1)
            rect = plt.Rectangle([-dist_to_weights - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
            if(n_asked_numbers==1):
                ax.add_patch(rect)
            
            extra_distance = 0.4
            if(output_names is not None):
                ax.text(-dist_to_weights-1/2-extra_distance, y+0.3, str(output_names[y]), fontsize=small_font_size,horizontalalignment='right')
            

            
    for (x, y), w in np.ndenumerate(matrix):
        color, size = get_rect_size_and_color(amplitude=w,max_amplitude=max_weight,amplitude_encoding=weight_encoding, fixed_size=1)    
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    
    # get lower corners of highlighted numbers
    if(highlight_number is not None):
        for n in highlight_number:
            highlight_lower_corners[n] = [n-1/2, +matrix.shape[1]- 1 / 2]

            # Highlight: highlight_number
            highlight_color = (1,1,0,0.2)
            highlight_border_color = (1,1,0,1.0)
            #(x1, y1), width=x4-x1, height=y2-y1, linewidth=1,
            extra_dist_2=0.15
            include_number=True
            extra_y_up=0.0
            if(include_number):
                extra_y_up=1.6 + extra_dist_3

            x4= highlight_upper_corners[n][0]+0.9+extra_dist_2-extra_dist_3/5
            y2= highlight_upper_corners[n][1]-extra_dist_2-extra_y_up
            x1= highlight_lower_corners[n][0]-extra_dist_2
            y1= highlight_lower_corners[n][1]+extra_dist_2
            rect = plt.Rectangle((x1, y1), width=x4-x1, height=y2-y1, linewidth=1.9,
                facecolor='none', edgecolor=highlight_border_color)
            ax.add_patch(rect)
    ### Highlight outputnumber        
    if(highlight_output_number is not None):
        for n in highlight_output_number:
            highlight_lower_corners[n] = [n-1/2, +matrix.shape[1]- 1 / 2]
            highlight_upper_corners[n] = [n-1/2, +matrix.shape[1]- 1 / 2]

            # Highlight: highlight_number
            highlight_color = (1,1,0,0.2)
            highlight_border_color = (1,1,0,1.0)
            #(x1, y1), width=x4-x1, height=y2-y1, linewidth=1,
            extra_dist_2=0.15
            include_number=True
            extra_y_up=0.0
            if(include_number):
                extra_y_up=1.6

            x4= matrix.shape[0]
            x1= 0-1/2-extra_dist_2-dist_to_weights-1-extra_distance-1
            y2= n + 1 / 2+extra_distance/2

            y1= n - 1 / 2-extra_distance/2
            rect = plt.Rectangle((x1, y1), width=x4-x1, height=y2-y1, linewidth=1.9,
                facecolor='none', edgecolor=highlight_border_color)
            ax.add_patch(rect)  
            
    # Add "W" description
    text_dist = 1
    text_rect = plt.Rectangle([matrix.shape[0]/2-2, matrix.shape[1]+text_dist+1], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)
    ax.text(matrix.shape[0]/2-2, matrix.shape[1]+text_dist+1, "W", fontsize=26)
    
    # Add representation description
    text_dist = 1
    text_rect = plt.Rectangle([matrix.shape[0]/2-2, -text_dist-dist_to_weights-size-3-n_tasks*n_asked_numbers-a*distance_between_asked_numbers], 1, 0.1, facecolor='gray', edgecolor='gray')
    ax.add_patch(text_rect)
    
    if(input_node_description=='Input'):
        input_node_description='Auditory feedback'
    ax.text(matrix.shape[0]/2-7.0-4, -text_dist-dist_to_weights-size-n_tasks*n_asked_numbers-a*distance_between_asked_numbers-1-extra_dist_3, input_node_description, fontsize=20) #30
        
    # Draw a horizontal line to separate Weight matrix and node activity:
    # draw vertical line from (70,100) to (70, 250)
    plt.plot([-2, matrix.shape[0]], [-1, -1], 'k-', lw=2)
    
    ax.autoscale_view()
    ax.invert_yaxis()

    
    return fig


########################
## Get graph from task trial
########################

def get_graph_from_task_trial(PATH, task, n_squares, model, env):

        env.reset()


        #PATH = "C:/Users/silvests/Embodied_counting/Results/count_all_objects__1_TIMES__7989/count_all_objects_1_to_9_20-03-13-12-51model-6645_/count_all_objects_1_to_9_20-03-13-12-51model-6645_"
        #PATH = "C:/Users/silvests/Embodied_counting/Results/master_all/master_all/multiple_tasks_1_to_9_20-03-31-01-29model-8138_"
        model.load_state_dict(torch.load(PATH))

        #save_network_path = RESULTS_PATH +"Network_Activity"
        #demonstrate_model(env, model, display_network_activity=True, save_network_path=save_network_path)
        demonstrate_model(env, model, display_=False, display_network_activity=False)
    
        return model.network_graph_list
    
task_to_readable_task = {
    'recite_n': 'Recite-N',
    'count_all_events': 'Count all Events',
    'count_all_objects': 'Count all Objects',
    'give_n': 'Give-N'
}

def save_hinton_diagram(graph_list, task,repr_type, ampl_encoding, filetype='pdf'):
        graphy = graph_list[0]
        matrix=graphy.weights[(0,1)].transpose()
        scale_fig=0.5
        fig_sizy=(matrix.shape[0]*scale_fig,matrix.shape[1]*scale_fig*len(graph_list))
        #fig, ax = plt.subplots(figsize=fig_sizy)
        fig, axs =  plt.subplots(len(graph_list), 1, figsize=fig_sizy)
        
        readable_task = task_to_readable_task[task]
        
        title_string = "Representation for solving task: " + readable_task
        #axs[0].text(0, -30, title_string, fontsize=25)
        #fig.title(title_string)

        for j in range(len(graph_list)):

            graphy = graph_list[j]

            weight_matrix=graphy.weights[(0,1)].transpose()
            input_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[0]])
            output_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[1]])
            output_names = np.asarray([graphy.txt[i] for i in graphy.nodes_in_layer[1]])
            #ampl_encoding = 'size'
            #ampl_encoding = 'transparency'


            fig = hinton(weight_matrix,figy=fig, axis=axs[j], input_nodes=input_nodes,output_nodes=output_nodes,output_names=output_names, weight_encoding=ampl_encoding, graphy=graphy)
        SUBDIR =  repr_type + '_repr_' + ampl_encoding + '_encoded/'
        directory_path = RESULTS_PATH + SUBDIR
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        PATH = directory_path + repr_type + '_repr_' + task + '_' + ampl_encoding + '_encoded.' + filetype
        fig.savefig(PATH)




from matplotlib.backends.backend_pdf import PdfPages

task_to_readable_task = {
    'recite_n': 'Recite-N',
    'count_all_events': 'Count all Events',
    'count_all_objects': 'Count all Objects',
    'give_n': 'Give-N'
}

repr_type_to_node_description = {
    'entity': 'Visual representation',
    'number_sequ': 'Language representation'
}

def save_hinton_diagram(graph_list, task,repr_type, ampl_encoding, filetype='pdf'):
        graphy = graph_list[0]
        matrix=graphy.weights[(0,1)].transpose()
        scale_fig=0.5
        
        nr_imgs_per_page = 5
        
        fig_sizy=(matrix.shape[0]*scale_fig,matrix.shape[1]*scale_fig*nr_imgs_per_page)
        #fig, ax = plt.subplots(figsize=fig_sizy)
        fig, axs =  plt.subplots(nr_imgs_per_page, 1, figsize=fig_sizy)
        
        readable_task = task_to_readable_task[task]
        
        title_string = "Representation for solving task: " + readable_task
        #axs[0].text(0.5*(left+right), -10, title_string, fontsize=25)
        axs[0].set_title(title_string, fontsize=25, pad=60)

        
        SUBDIR =  repr_type + '_repr_' + ampl_encoding + '_encoded/'
        directory_path = RESULTS_PATH + SUBDIR
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        PATH = directory_path + repr_type + '_repr_' + task + '_' + ampl_encoding + '_encoded.' + filetype
        
        with PdfPages(PATH) as pdf:
                for j in range(len(graph_list)):

                    graphy = graph_list[j]

                    weight_matrix=graphy.weights[(0,1)].transpose()
                    input_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[0]])
                    output_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[1]])
                    output_names = np.asarray([graphy.txt[i] for i in graphy.nodes_in_layer[1]])
                    #ampl_encoding = 'size'
                    #ampl_encoding = 'transparency'
                    
                    node_description = repr_type_to_node_description[repr_type]
                    fig = hinton(weight_matrix,
                                 figy=fig, 
                                 axis=axs[j%5], 
                                 input_nodes=input_nodes,
                                 input_node_description=node_description,
                                 output_nodes=output_nodes,
                                 output_names=output_names, 
                                 weight_encoding=ampl_encoding, graphy=graphy)
                                        
                    ##### create a new figure and a new page every 5 imgs:
                    if(j%nr_imgs_per_page==nr_imgs_per_page-1 or j==len(graph_list)-1):
                        #fig.savefig(PATH)
                        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
                        plt.close()
                        fig, axs =  plt.subplots(nr_imgs_per_page, 1, figsize=fig_sizy)

def hinton_from_graph_only(graphy, fig, ax):
        #graphy = graph_list[j]

        weight_matrix=graphy.weights[(0,1)].transpose()
        input_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[0]])
        output_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[1]])
        output_names = np.asarray([graphy.txt[i] for i in graphy.nodes_in_layer[1]])
        #ampl_encoding = 'size'
        #ampl_encoding = 'transparency'

        node_description = repr_type_to_node_description[repr_type]
        fig = hinton(weight_matrix,
                     figy=fig, 
                     axis=ax, 
                     input_nodes=input_nodes,
                     input_node_description=node_description,
                     output_nodes=output_nodes,
                     output_names=output_names, 
                     weight_encoding=ampl_encoding, graphy=graphy)
        return fig
    
def get_correct_time_steps(graph_list, output_indices_on):
    
        correct_time_steps = []
        for j in range(len(graph_list)):

            graphy = graph_list[j]

            #weight_matrix=graphy.weights[(0,1)].transpose()
            last_layer = len(graphy.nodes_in_layer)
            input_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[0]])
            output_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[last_layer-1]])

            if(type(output_indices_on[0])==int):
                for i in output_indices_on:
                    if output_nodes[i]>0.8:
                        correct_time_steps.append(j)

            if(output_indices_on[0]=='no_count'):
                not_counting = True
                for i in range(0,10):
                    if(output_nodes[i]>0.8):
                        not_counting=False
                if(not_counting):
                   correct_time_steps.append(j)
        return correct_time_steps
    
    
def get_time_steps_for_count_and_no_count(graph_list):
    time_steps_count = {}

    for i in range(0,11):    
        time_steps_count[i] = get_correct_time_steps(graph_list, [i])
    time_steps_no_count = get_correct_time_steps(graph_list, ['no_count'])

    return time_steps_count, time_steps_no_count



class GraphListClass():
    def __init__(self, graph_list, task):
        self.graph_list = graph_list
        self.n_time_steps = len(graph_list)
        self.task = task
        self.time_steps_count, self.time_steps_no_count = get_time_steps_for_count_and_no_count(graph_list)
        self.time_steps_count_and_no_count = {}
        for i in range(11):            
            self.time_steps_count_and_no_count[i] = self.time_steps_count[i]
        self.time_steps_count_and_no_count['no_count'] = self.time_steps_no_count
        
        # Get all nodes layerwise
        self.node_activations = []
        
        for l in range(len(graph_list[0].nodes_in_layer)): 
        
            node_act_for_all_layers_at_time_t = []
            
            for t in range(len(graph_list)):
                # later graph_list[0].n_layers   
                graphy=graph_list[t]
                node_activations = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[l]])
                node_act_for_all_layers_at_time_t.append(node_activations)
            self.node_activations.append(node_act_for_all_layers_at_time_t)
            
        self.input_nodes_list, self.output_nodes_list = self.get_input_and_output_nodes_from_graph_list()
        #self.weight_matrix_list = self.get_weight_matrix()
        graphy = self.graph_list[0]
        #self.output_names = np.asarray([graphy.txt[i] for i in graphy.nodes_in_layer[1]])
        self.output_node_description = self.graph_list[0].layer_descriptions[1]
        self.input_node_description = self.graph_list[0].layer_descriptions[0]
        
    def get_input_and_output_nodes_from_graph_list(self):
        
        input_nodes_list = []        
        output_nodes_list = []
        
        for t in range(self.n_time_steps):
            graphy = self.graph_list[t]
            input_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[0]])
            output_nodes = np.asarray([graphy.node_id_activation[i] for i in graphy.nodes_in_layer[1]])
            input_nodes_list.append(input_nodes)
            output_nodes_list.append(output_nodes)
        
        return input_nodes_list, output_nodes_list
        
    def get_weight_matrix(self):
        weight_matrix_list = []
        for i in range(self.n_time_steps):
            weight_matrix=self.graph_list[i].weights[(0,1)].transpose()
            weight_matrix_list.append(weight_matrix)
        return weight_matrix_list
        

def compare_neural_activities(asked_numbers):
    graphy = graph_list[0]
    scale_fig=0.5
    matrix=graphy.weights[layers].transpose()
    
    n_rows = len(asked_numbers)
    
    fig_sizy=(matrix.shape[0]*scale_fig*n_cols,matrix.shape[1]*scale_fig*n_rows)

    fig, axs =  plt.subplots(n_rows, n_cols, figsize=fig_sizy)
    fig.tight_layout(pad=0.01)
    #fig.subplots_adjust(left=0.00, bottom=None, right=0.01, top=None, wspace=None, hspace=None)
    fig.subplots_adjust(right=0.6)
    
    #readable_task = task_to_readable_task[task]
    #axs[0].text(0.5*(left+right), -10, title_string, fontsize=25)    
    title_string = "Representation for solving task: Count all events"     
    axs[0,0].set_title(title_string, fontsize=25, pad=60)
    title_string = "Representation for solving task: Count all objects"     
    axs[0,1].set_title(title_string, fontsize=25, pad=60)
    title_string = "Representation for solving task: Give-N"     
    axs[0,2].set_title(title_string, fontsize=25, pad=60)
    i=0
    for n in asked_numbers:
        if(type(n)==int):
            fig = hinton_from_graph_only(graph_count_events[time_steps_count_events[n][0]], fig, axs[i,0])
            fig = hinton_from_graph_only(graph_count_objects[time_steps_count_objects[n][0]], fig, axs[i,1])
            fig = hinton_from_graph_only(graph_give_n[time_steps_count_give_n[n][0]], fig, axs[i,2])
        if(n=='no_count'):
            fig = hinton_from_graph_only(graph_count_events[time_steps_no_count_events[0]], fig, axs[i,0])
            fig = hinton_from_graph_only(graph_count_objects[time_steps_no_count_objects[0]], fig, axs[i,1])
            fig = hinton_from_graph_only(graph_give_n[time_steps_no_count_give_n[0]], fig, axs[i,2])
        i+=1   

        
def compare_neural_activities(multiple_graph_lists, asked_numbers, layers=(0,1)):
    graphy = multiple_graph_lists[0].graph_list[0]
    scale_fig=0.5
    matrix=graphy.weights[layers].transpose()
    
    n_rows = len(asked_numbers)
    n_cols = len(multiple_graph_lists)
    
    fig_sizy=(matrix.shape[0]*scale_fig*n_cols,matrix.shape[1]*scale_fig*n_rows)

    fig, axs =  plt.subplots(n_rows, n_cols, figsize=fig_sizy)
    fig.tight_layout(pad=0.01)
    #fig.subplots_adjust(left=0.00, bottom=None, right=0.01, top=None, wspace=None, hspace=None)
    fig.subplots_adjust(right=0.6)
    
    #readable_task = task_to_readable_task[task]
    #axs[0].text(0.5*(left+right), -10, title_string, fontsize=25)    
    for tasky in range(len(multiple_graph_lists)):
        title_string = "Representation for solving task: " + multiple_graph_lists[tasky].task     
        axs[0,tasky].set_title(title_string, fontsize=25, pad=60)

    i=0
    for n in asked_numbers:
        if(type(n)==int):
            for tasky in range(len(multiple_graph_lists)):
                graph_list = multiple_graph_lists[tasky].graph_list
                time_step =  multiple_graph_lists[tasky].time_steps_count[n][0]              
                fig = hinton_from_graph_only(graph_list[time_step], fig, axs[i,tasky], layers=layers)
        if(n=='no_count'):
            for tasky in range(len(multiple_graph_lists)):
                graph_list = multiple_graph_lists[tasky].graph_list
                time_step =  multiple_graph_lists[tasky].time_steps_no_count[0]              
                fig = hinton_from_graph_only(graph_list[time_step], fig, axs[i,tasky], layers=layers)
        i+=1  
        
        
