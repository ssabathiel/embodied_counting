# Define some constants
KERNEL_SIZE = 3
PADDING = 1 #KERNEL_SIZE // 2

print("Loading model....???")

class LangConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, env,lr):
        super().__init__()
        
        self.model_id = random.randint(1,10000) 
        self.episode = 0
        self.lr = lr
        self.result_tensory = result_tensor()
        self.n_solved_for_m = None
        self.time_run = 0.0
        self.model_path = ""
        self.test_mode = False
        
        img_size = env.view_size
        n_count_words = env.n_words
        n_motor_actions = env.n_motor_actions
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_count_words = n_count_words
        self.task = env.task_n
        self.task_vector_length = env.task_vector_length        
        self.draw_graphy = False
        self.network_graph_list = []
        n_actions = 4
                
        ## Define number of hidden neurons (attention: some are restricted)
        self.vis_representation_size = 60 #60
        self.memory_length = 30*(n_count_words+1) #6*(n_count_words+1)
        self.vis_lang_representation_size = self.vis_representation_size  + self.memory_length
        self.lang_representation_size = self.vis_lang_representation_size
        
        #########################
        ## Define LSTM-Gates
        ########################
        # Conv-LSTM:
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)
        # FC-LSTM:        
        #self.Gates_lang = nn.Linear((n_count_words+1)+self.memory_length+self.task_vector_length, 3*self.memory_length )
        #self.Gates_lang_forget = nn.Linear((n_count_words+1)+self.memory_length+self.task_vector_length, self.memory_length )
        #self.Gates_lang_forget.bias.data.fill_(1.0)
        lang_input_size = (n_count_words+1)+self.memory_length+self.task_vector_length
        self.Gates_lang = torch.nn.LSTM(input_size=lang_input_size,hidden_size=3*self.memory_length, num_layers=1)
        self.Gates_lang_forget = torch.nn.LSTM(input_size=lang_input_size,hidden_size=self.memory_length, num_layers=1)
        self.lang_mem_forget_states = None
        self.lang_mem_states = None
        ########################
        ## Define fully connected Layers
        ##############################
        
        ## Representations
        # From flattened-2D to vis-representation        
        self.fc1 = nn.Linear(img_size*img_size*hidden_size+self.task_vector_length, self.vis_representation_size)
		    # From vis-lang-task representation to lang-representation
        self.fc3_lang = nn.Linear(self.vis_lang_representation_size, self.lang_representation_size)        
        ## Outputs: Motor-action, Verbal-action, IsDoMotoraction, IsSayWord,
        # Motor-action
        self.fc2 = nn.Linear(self.vis_lang_representation_size, n_motor_actions)
        # Word
        self.fc4_lang = nn.Linear(self.lang_representation_size, n_count_words)        
        #IsSayWord
        self.fc5_lang = nn.Linear(self.lang_representation_size, 1)
        #IsDoMotoraction
        self.fc6 = nn.Linear(self.vis_lang_representation_size, 1)        
                
        #################
        ## Declare layers, such they are accessible from the outside
        ########################################
        self.cell = None
        self.hidden = None
        self.cell_lang = None
        self.hidden_lang = None
        self.vis_representation_layer = None
        self.output_lang_repr = None


    def forward(self, input_, prev_state, input_lang, prev_state_lang, env_task_list):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if(CUDA_bool==False):
           if prev_state is None:
               state_size = [batch_size, self.hidden_size] + list(spatial_size)
               prev_state = (
                   Variable(torch.zeros(state_size)),  #.cuda()
                   Variable(torch.zeros(state_size))   #.cuda()
               )
           if(prev_state_lang is None):
               state_size = [batch_size, self.memory_length]
               prev_state_lang = (
                   Variable(torch.zeros(state_size)),  #.cuda()
                   Variable(torch.zeros(state_size))  #.cuda()
               )
               lang_mem_state_size = [1, batch_size, 3*self.memory_length]
               self.lang_mem_states = (
                   Variable(torch.zeros(lang_mem_state_size)),  #.cuda()
                   Variable(torch.zeros(lang_mem_state_size))   #.cuda()
               )
               lang_mem_forget_state_size = [1, batch_size, self.memory_length]
               self.lang_mem_forget_states = (
                  Variable(torch.zeros(lang_mem_forget_state_size)),  #.cuda()
                  Variable(torch.zeros(lang_mem_forget_state_size))   #.cuda()
               )
        else:
           if prev_state is None:
               state_size = [batch_size, self.hidden_size] + list(spatial_size)
               prev_state = (
                   Variable(torch.zeros(state_size)).cuda(),  #.cuda()
                   Variable(torch.zeros(state_size)).cuda()   #.cuda()
               )
           if(prev_state_lang is None):
               state_size = [batch_size, self.memory_length]
               prev_state_lang = (
                   Variable(torch.zeros(state_size)).cuda(),  #.cuda()
                   Variable(torch.zeros(state_size)).cuda()  #.cuda()
               )
               lang_mem_state_size = [1, batch_size, 3*self.memory_length]
               self.lang_mem_states = (
                   Variable(torch.zeros(lang_mem_state_size)),  #.cuda()
                   Variable(torch.zeros(lang_mem_state_size))   #.cuda()
               )
               lang_mem_forget_state_size = [1, batch_size, self.memory_length]
               self.lang_mem_forget_states = (
                   Variable(torch.zeros(lang_mem_forget_state_size)),  #.cuda()
                   Variable(torch.zeros(lang_mem_forget_state_size))   #.cuda()
               )
        #################################
        ## Conv-LSTM
        #################################
        prev_hidden, prev_cell = prev_state
        #input_ = input_*5
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
     
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        output = hidden.view(batch_size, -1 )
        stacked_output_and_task_vector = torch.cat((output, env_task_list), 1)
        
        output_repr = f.relu(self.fc1(stacked_output_and_task_vector))
        if(self.test_mode == False):
            drops = nn.Dropout(p=0.4)
            output_repr = drops(output_repr)
        #output_repr = output
        

        #################################
        ## LSTM
        #################################
        
        prev_hidden_lang, prev_cell_lang = prev_state_lang
        stacked_inputs_lang = torch.cat((input_lang, prev_hidden_lang, env_task_list), 1).view(-1, (self.n_count_words+1)+self.memory_length+self.task_vector_length )
        
        #gates_lang = self.Gates_lang(stacked_inputs_lang) 
        #gates_lang_forget = self.Gates_lang_forget(stacked_inputs_lang)
        gates_lang, self.lang_mem_states = self.Gates_lang(stacked_inputs_lang.unsqueeze(dim=0), self.lang_mem_states) 
        gates_lang = gates_lang[-1]
        gates_lang_forget, self.lang_mem_forget_states = self.Gates_lang_forget(stacked_inputs_lang.unsqueeze(dim=0), self.lang_mem_forget_states) 
        gates_lang_forget = gates_lang_forget[-1]

        in_gate_lang, out_gate_lang, cell_gate_lang = gates_lang.chunk(3, 1)
        remember_gate_lang = gates_lang_forget
        cell_gate_lang = torch.tanh(cell_gate_lang)
        in_gate_lang = torch.sigmoid(in_gate_lang)
        remember_gate_lang = torch.sigmoid(remember_gate_lang)
        out_gate_lang = torch.sigmoid(out_gate_lang)        
        
        
        cell_lang = (remember_gate_lang * prev_cell_lang) + (in_gate_lang * cell_gate_lang)
        hidden_lang = out_gate_lang * torch.tanh(cell_lang)
        output_lang = hidden_lang.view(batch_size, -1 )
        if(self.test_mode == False):
            drops = nn.Dropout(p=0.4)
            output_lang = drops(output_lang)
        

 
          
		    ######################################
    	  ## Joined Vis-Lang-Task representation
        stacked_lang_vis_repr = torch.cat((output_lang, output_repr), 1)
         
        ######################################
        ## Motor action
        IsDoMotorAction = torch.sigmoid(self.fc6(stacked_lang_vis_repr))        
        output = torch.sigmoid(self.fc2(stacked_lang_vis_repr))        
        stacked_output = torch.cat((output, IsDoMotorAction), 1)
        
        ######################################
        ## Verbal action          
        #output_lang_repr = f.relu(self.fc3_lang(stacked_lang_vis_repr))
        output_lang_repr = stacked_lang_vis_repr
        output_lang = torch.sigmoid(self.fc4_lang(output_lang_repr))
        output_isCount = torch.sigmoid(self.fc5_lang(output_lang_repr))        
        stacked_output_lang = torch.cat((output_lang, output_isCount), 1)
        
        #################
        ## Define layers, such they are accessible from the outside
        ########################################
        self.cell = cell
        self.hidden = hidden
        self.cell_lang = cell_lang
        self.hidden_lang = hidden_lang
        
        self.vis_representation_layer = output_repr
        self.output_lang_repr = output_lang_repr
        
        
        if(self.draw_graphy==True):
            #print("in draw graphy")
            G=nx.DiGraph()
            self.own_G = own_G(G)
            #print(input_stacked.shape)
            node_description_list_ = ["C-A", "C-B", "A", "B", "End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            output_description_list = ["End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            add_layer(self.own_G, 0, activations = input_stacked.tolist(),node_description_list=node_description_list_, description_pos='middle left', layer_description='Input')
            
            add_layer(self.own_G, 1, activations = remember_gate.tolist(),layer_description='Remember-gate')
            add_layer(self.own_G, 2, activations = prev_cell.tolist(),layer_description='Prev Cell', layer_operation='x')
            add_layer(self.own_G, 3, activations = in_gate.tolist(),layer_description='In-Gate', layer_operation='+')
            add_layer(self.own_G, 4, activations = cell_gate.tolist(),layer_description='Cell-Gate', layer_operation='x')
            
            add_layer(self.own_G, 5, activations = torch.tanh(self.cell).tolist(),layer_description='tanh(Cell)', layer_operation='=')
            add_layer(self.own_G, 6, activations = out_gate.tolist(),layer_description='Out gate', layer_operation='x')
            
            add_layer(self.own_G, 7, activations = torch.tanh(self.hidden).tolist(),layer_description='tanh(hidden)', layer_operation='x')
            add_layer(self.own_G, 8, activations = self.stacked_output_lang.tolist(),node_description_list=output_description_list, description_pos='middle right',layer_description='Output')
            
            #print("self.Gates.weight.data.detach().shape: ", self.Gates.weight.data.detach().shape)
            #weights_i = self.Gates.weight.data.detach().numpy()[:self.memory_size,:]
            #weights_r = self.Gates.weight.data.detach().numpy()[1*self.memory_size:2*self.memory_size+1,:]
            #weights_o = self.Gates.weight.data.detach().numpy()[2*self.memory_size:3*self.memory_size+1,:]
            #weights_c = self.Gates.weight.data.detach().numpy()[3*self.memory_size:4*self.memory_size+1,:]
            weights_i, weights_r, weights_o,weights_c = np.split(self.Gates.weight.data.detach().numpy(), 4,axis=0)
            
            #print("self.Gates.weight.data.detach().numpy().shape: ",self.Gates.weight.data.detach().numpy().shape)
            #print("self.Gates.weight.data.detach().numpy()[:,1*self.memory_size+1:2*self.memory_size].shape: ",self.Gates.weight.data.detach().numpy()[1*self.memory_size+1:2*self.memory_size+1,:].shape)
            
            
            weights_2 = self.fc1.weight.data.detach()
            weights_3 = self.fc3.weight.data.detach()
            weights_4 = self.fc2.weight.data.detach()
            weights_5 = self.fc4.weight.data.detach()

            stacked_weights = torch.cat((weights_2, weights_3,weights_4,weights_5), 0).numpy()
            #print("stacked_weights.shape: ", stacked_weights.shape)
            
            #weights_1 = self.fc1.weight.data.detach().numpy()
            #weights_2 = self.fc2.weight.data.detach().numpy()
            
            connect_layer(self.own_G, 0, 1, weights = weights_r)
            connect_layer(self.own_G, 0, 3, weights = weights_i)
            connect_layer(self.own_G, 0, 4, weights = weights_c)
            connect_layer(self.own_G, 0, 6, weights = weights_o)
            
            connect_layer(self.own_G, 7, 8, weights = stacked_weights)
            
            self.network_graph_list.append(self.own_G)
        
        
        
        return [hidden, cell], stacked_output, [hidden_lang, cell_lang], stacked_output_lang

      
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features