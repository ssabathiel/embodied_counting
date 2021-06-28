
print("Load own Graph class ...")
import numpy as np
import matplotlib.pyplot as plt

######### Graph visualization
class own_G():
  def __init__(self, G):
      self.global_id = 0
      self.node_id_activation = {}
      self.nodes_in_layer = {}
      self.activation_in_layer = {}
      self.weights = {}
      self.G = G
      self.pos = {}
      self.txt = {}
      self.txt_pos = {}
      self.activations = {}
      self.layer_descriptions = {} #[layer_description_1, layer_description_2,layer_description_3 ]
      self.layer_operation = {}
      self.n_layers = 0
      


def bipolar_number_to_color(numby):
  if(numby<=0):
    colory = (1, 0, 0,round( abs(numby),2 ) )
  else:
    colory = (0, 0, 1,round( abs(numby),2 ) )
  return colory

def bipolar_number_to_color_255(numby):
  if(numby<=0):
    colory = (255, 0, 0,round( abs(numby),2 ) )
  else:
    colory = (0, 0, 255,round( abs(numby),2 ) )
  return colory

def add_layer(own_G, layer_nr, activations, node_description_list=None, description_pos='middle left', layer_description='', layer_operation=None ):
    sub_nodelist = []
    alpha_list = []
    
    own_G.n_layers += 1
    
    activations = activations[0]
    scaly_normalizer = len(activations)/15
    scaly = 0.6 #0.03*scaly_normalizer
    
    if(layer_nr==0):
        x_of_last_layer = 0
    else:
        x_of_last_layer = own_G.pos[own_G.nodes_in_layer[layer_nr-1][0]][0]
    
    last_pos_in_layer = [0,0]
    
    layer_distance = 10
    if(layer_operation is not None):
        
        if(layer_operation=='x'):
            layer_distance = 7
        operation_y = (scaly*len(activations))/2
        operation_pos = [x_of_last_layer+(layer_distance/2), operation_y]
        own_G.layer_operation[layer_nr] = (layer_operation, operation_pos)
    
    for i in range(len(activations)):
        node_name = own_G.global_id
        own_G.global_id += 1
        own_G.G.add_node(node_name)
        
        y = scaly*i
        x = x_of_last_layer+layer_distance
        own_G.pos[node_name] = [np.asarray(x), y]   #x was layer_nr*10
        last_pos_in_layer = [np.asarray(x), y + scaly]
        
        own_G.activations[node_name] = activations[i]
        own_G.txt_pos[node_name] = description_pos
        
        if(node_description_list is not None):
            own_G.txt[node_name] = node_description_list[i]
             
        sub_nodelist.append(node_name)
        
        if(type(activations)==list):
          own_G.node_id_activation[node_name] = round( activations[i] ,2 )
        else:
          own_G.node_id_activation[node_name] = "0.55"
          
        alpha_list.append( node_name/14 )
    
    own_G.layer_descriptions[layer_nr] = (layer_description, last_pos_in_layer)
    own_G.nodes_in_layer[layer_nr] = sub_nodelist  
    plt.figure(1,figsize=(12,12)) 
    colors = list( map(bipolar_number_to_color, activations ) )
    
    #nx.draw_networkx_nodes(own_G.G,own_G.pos, nodelist=sub_nodelist, node_color=colors,node_size=800 )
    #nx.draw_networkx_labels(own_G.G,own_G.pos, labels=own_G.node_id_activation, nodelist=sub_nodelist) 
    #nodes = (own_G.G,own_G.pos, nodelist=sub_nodelist, node_color=colors,node_size=800 )
    #nx.draw_networkx_labels(own_G.G,own_G.pos, labels=own_G.node_id_activation, nodelist=sub_nodelist) 
    #if(node_description_list is not None):
    #    nx.draw_networkx_labels(own_G.G,own_G.pos, labels=node_description_list, nodelist=sub_nodelist) 
    #nx.draw(own_G,pos, with_labels=True,  nodelist=sub_nodelist) #nodelist=sub_nodelist
        
def connect_layer(own_G, layer_1, layer_2, line_style=None, weights = 0.5):  
      edge_listy = []
      width_list = []
      label_list = []
      edge_labels = {}
      weight_name = (layer_1, layer_2)
      own_G.weights[weight_name] = weights
      
      
      #print( "len(own_G.nodes_in_layer[layer_1])", len(own_G.nodes_in_layer[layer_1]) )
      #print( "len(own_G.nodes_in_layer[layer_2])", len(own_G.nodes_in_layer[layer_2]) )
      #print("weights.shape: ", weights.shape)
      n=0  
      for i in own_G.nodes_in_layer[layer_1]: 
          m = 0
          for j in own_G.nodes_in_layer[layer_2]:              
              own_G.G.add_edge(i, j, weight=weights[m,n])
              edge_listy.append([i, j])
              #width_list.append(i)
              if(type(weights)!=float):
                width_list.append(weights[m,n])
                #print("add weights: ", weights[m,n])
              else:
                width_list.append(i)
              labelly = str(i) + "-" + str(j)
              edge_labels[(i,j)] = labelly
              m+=1
          n += 1   
      #arrowstyle = patches.ArrowStyle("Fancy, head_length=1.0, head_width=.4, tail_width=.10") #Fancy/Simple
      #nx.draw_networkx_edges(own_G.G,own_G.pos, connectionstyle=line_style, edgelist = edge_listy, width=width_list, arrowstyle = arrowstyle, edge_labels=edge_labels,node_size=800 )
      # arrowstyle = patches.ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")