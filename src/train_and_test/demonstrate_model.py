#################################
###### DEMONSTRATION
###########################
from PIL import Image
from IPython.display import display, update_display
import torch
from train_and_test.env_to_pytorch_interface import *
from copy import copy
CUDA_bool = False
import time


print("Import demonstrate-model ..")

def demonstrate_model(env, model, PATH=None, display_=False, display_network_activity=False, save_network_path=None):
  
      #print("in demonstrate..")   
      model.training = False
      state_network = None

      env.rand_n_squares = False
      env.save_epoch = True
      
      env.reset()
      if(display_):
        display(env.observationImg.convert('RGB'), display_id = "game")
        update_display(env.observationImg.convert('RGB'), display_id = "game")

      delay_time = 0.20

      image_list = []
      network_image_list = []
      cell_list = []
      hidden_list = []

      env.reset()
      max_t = env.max_time
      t=0
      state_network = None
      n=0
      task_vector_size = 5

      state_network_vis = None
      state_network_lang = None
      input_lang = torch.zeros( (1, env.n_words+1) ).view(1, env.n_words+1)
      task_vector = torch.zeros(task_vector_size)
      task_vector[env.task_n] = 1

      task_vector = create_task_vector(env)
      task_vector = task_vector.reshape(1, env.task_vector_length)
      
      a_is = []

      while(t<max_t and env.ended == False):
          '''
          t += 1

          image, action = envImageAndActionToPytorchFormat(env)
          #stacked_img_coord = add_coordinate_layers(image, env.img_size, env.task_n)
          state_network, Q_values = model.forward(image, state_network)
          a = np.argmax(Q_values.detach().numpy()).item()
          print(a)
          env.update(a)
          '''

          ################NEW PART END      

          #if(n==0):
          #    env.display = "Game"
          t += 1

          image, action = envImageAndActionToPytorchFormat(env)
          #stacked_img_coord = add_task_layer(image, env.img_size, env.task_n)
          #stacked_img_coord = add_task_layer(image, env.img_size, env)
          stacked_img_coord = image
          task_vector = create_task_vector(env)
          #state_network, Q_values = model.forward(stacked_img_coord, state_network)
          state_network_vis, output_action, state_network_lang, output_lang = model(stacked_img_coord,state_network_vis, input_lang, state_network_lang, task_vector)
          #print("###########################")
          #print("whole action: ", torch.cat((output_action, output_lang),1))
          #print("output_lang.detach().numpy(): ", output_lang.detach().numpy() )
          #print("output_action.detach().numpy(): ", output_action.detach().numpy() )
          #print("output_lang.detach().numpy()[0][:-1]: ", output_lang.detach().numpy()[0][:-1] )
          #print("output_lang[0][-1].detach().numpy(): ", output_lang[0][-1].detach().numpy())


          Q_values = torch.cat((output_action, output_lang),1)
          #print(Q_values[0].tolist())

          a = int(np.argmax(output_action.detach().numpy()[0][:-1]).item() )
          Is_a = bool( round( output_action[0][-1].detach().numpy().item() ) )

          #print(output_lang.detach().numpy()[0][-1])
          c = bool( round( output_lang[0][-1].detach().numpy().item() ) )


          word = int(np.argmax(output_lang.detach().numpy()[0][:-1]).item() )

          triple_action_arr = np.concatenate((output_action.detach().numpy(),output_lang.detach().numpy()), axis=None)
          #print(np.around(triple_action_arr,decimals=2))
          #print("Motor-Action:" , np.around(output_action.detach().numpy(),decimals=2))
          #print("Verbal-Action:" , np.around(output_lang.detach().numpy(),decimals=2))

          
          #if(c):
          #    print(word+1)
          #    print(state_network_lang)

          #print("word: ", word)
          #print("a: ", Action[a]  )
          #print("word: ", word)
          #print("Action[word]: ", Action[word])
          #print("c: ", c)
          #print("int(a): ", int(a))
          env.triple_update(int(a), Is_a, int(word+env.n_motor_actions),c )


          #print("c: ", c)
          #c = True
          input_lang = torch.zeros( env.n_words+1 ) #.view(1, env.n_words+1)
          #if(c==True):
          #  input_lang[word-4] = 1
          #  input_lang[env.n_words] = 1

          #print("input_lang: ", input_lang)
          #input_lang = input_lang.view(1, env.n_words+1)
          input_lang = copy(output_lang)
          input_lang = torch.from_numpy( env.action_onehot[-env.n_words-1:] ).float().view(1, -1)
          
          
          
          ##### get variability:
          action_length = output_action.detach().numpy()[0][:-1].size
          verbal_length = output_lang.detach().numpy()[0][:-1].size
          n_actions = action_length + verbal_length
          #print("n_actions:", n_actions)
          
          if(Is_a):
          	a_is.append(a)
          if(c):
            a_is.append(word)


          ################NEW PART END      


          if(display_):
            
            img_input = env.observationImg.convert('RGB')
            #img_memory = Image.fromarray(state_network_vis[0][0][0].detach().numpy()*255).resize( (400,400), resample=0).convert('RGB')
            
            #imgs = get_concat_h(img_input, img_memory)
            imgs = img_input
            
            update_display(imgs, display_id = "game")
            #update_display(img_memory, display_id = "cell")
            time.sleep(delay_time*2)

          image_list.append(env.observationImg.convert('RGB'))

          
          if(display_network_activity):
            #########################
            ## Get layer-activities
            ########################
            #cell = self.cell
            #hidden = self.hidden
            
            task_vector_ = create_task_vector(env)
            cell_lang = model.cell_lang  
            hidden_lang = model.hidden_lang         
            vis_representation_layer = model.vis_representation_layer 
            output_lang_repr = model.output_lang_repr
            
            #########################
            ## Layer-activities --> to image
            ########################
            #cell_img = network_array_to_image(cell, img_width=45)
            #hidden_img = network_array_to_image(hidden, img_width=45)
            border_width = 5
            border_color = 'red'
            
            img_2D_size = 200
            img_2D_size_hidden = 100
            img_1D_size = 22
            img_1D_size_small = 15
            
            visual_input = env.observationImg.resize( (img_2D_size,img_2D_size), resample=0).convert('RGB')
            visual_input = ImageOps.expand(visual_input, border=border_width, fill = border_color)
            hidden_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 30)
            layer_description_img = Image.new('RGB', (visual_input.width, visual_input.width//6), color='white')
            draw = ImageDraw.Draw(layer_description_img)
            wordy = "Visual Input"
            w, h = draw.textsize(wordy, font=hidden_font)
            W = layer_description_img.width
            H = layer_description_img.height
            draw.text(((W-w)/2,(H-h)/2),wordy,(0,0,0), font=hidden_font)
            visual_input = get_concat_v(layer_description_img, visual_input, distance = 0)
            
            
            node_names_list = ["Touch", "Count", "Give", "Recite","Nothing","Objects", "Events","-", "-", "-", "ALL", "1", "2","3", "4","5", "6","7", "8","9"]
            layer_description = "TASK VECTOR"
            task_image = network_array_to_image(array_=task_vector_,node_names_list=node_names_list, layer_description=layer_description, img_width=img_1D_size)
            task_image_small = network_array_to_image(array_=task_vector_,node_names_list=node_names_list, layer_description=layer_description, img_width=img_1D_size_small)

            # Cell images 
            hidden_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 30)
            hidden_color = (200,200,250)

            n_hidden_layer = state_network_vis[0][0].detach().numpy().shape[0]
            cell_img = Image.fromarray(state_network_vis[0][0][0].detach().numpy()*255).resize( (img_2D_size_hidden,img_2D_size_hidden), resample=0).convert('RGB')
            cell_img = ImageOps.expand(cell_img, border=border_width, fill = border_color)
            for i in range(1,n_hidden_layer):
                next_cell_img = Image.fromarray(state_network_vis[0][0][i].detach().numpy()*255).resize( (img_2D_size_hidden,img_2D_size_hidden), resample=0).convert('RGB')
                next_cell_img = ImageOps.expand(next_cell_img, border=border_width, fill = border_color)
                cell_img = get_concat_h(cell_img, next_cell_img, distance = 50, colory=hidden_color)

            layer_description_img = Image.new('RGB', (cell_img.width, cell_img.width//6), color=hidden_color)
            draw = ImageDraw.Draw(layer_description_img)
            wordy = "ConvLSTM - Cells"
            w, h = draw.textsize(wordy, font=hidden_font)
            W = layer_description_img.width
            H = layer_description_img.height
            draw.text(((W-w)/2,(H-h)/2),wordy,(0,0,0), font=hidden_font)
            cell_img = get_concat_v(layer_description_img, cell_img, distance = 0, colory=hidden_color)
            #### END cell images
            
            # Hidden images 

            n_hidden_layer = state_network_vis[1][0].detach().numpy().shape[0]
            hidden_img = Image.fromarray(state_network_vis[1][0][0].detach().numpy()*255).resize( (img_2D_size_hidden,img_2D_size_hidden), resample=0).convert('RGB')
            hidden_img = ImageOps.expand(hidden_img, border=border_width, fill = border_color)
            for i in range(1,n_hidden_layer):
                next_cell_img = Image.fromarray(state_network_vis[1][0][i].detach().numpy()*255).resize( (img_2D_size_hidden,img_2D_size_hidden), resample=0).convert('RGB')
                next_cell_img = ImageOps.expand(next_cell_img, border=border_width, fill = border_color)
                hidden_img = get_concat_h(hidden_img, next_cell_img, distance = 50, colory=hidden_color)

            layer_description_img = Image.new('RGB', (hidden_img.width, hidden_img.width//6), color=hidden_color)
            draw = ImageDraw.Draw(layer_description_img)
            wordy = "ConvLSTM - Hidden"
            w, h = draw.textsize(wordy, font=hidden_font)
            W = layer_description_img.width
            H = layer_description_img.height
            draw.text(((W-w)/2,(H-h)/2),wordy,(0,0,0), font=hidden_font)
            hidden_img = get_concat_v(layer_description_img, hidden_img, distance = 0, colory=hidden_color)

            
            #### END cell images
                     
            cell_lang_img = network_array_to_image(cell_lang,layer_description="Cell LSTM-state\n (long-term memory)", img_width=img_1D_size)
            hidden_lang_img = network_array_to_image(hidden_lang, layer_description="Hidden LSTM-state\n (short-term memory)",  img_width=img_1D_size)
            hidden_lang_img_small = network_array_to_image(hidden_lang, layer_description="Hidden LSTM-state\n (short-term memory)",  img_width=img_1D_size_small)
            vis_representation_layer_img = network_array_to_image(vis_representation_layer, layer_description="Visual representation",  img_width=img_1D_size)
            vis_representation_layer_img_small = network_array_to_image(vis_representation_layer, layer_description="Visual representation",  img_width=img_1D_size_small)
            output_lang_repr_img = network_array_to_image(output_lang_repr, layer_description="Verbal representation", img_width=img_1D_size)
            output_lang_repr_img_small = network_array_to_image(output_lang_repr, layer_description="Verbal representation", img_width=img_1D_size_small)
            
            action_node_names = ["D", "U", "R", "L", "Pick", "Drop", "Touch","If-A", "End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            motor_action_node_names = ["D", "U", "R", "L", "Pick", "Drop", "Touch","If-A"]
            verbal_action_node_names = ["End/Wait", "1", "2", "3","4","5","6","7","8","9","If-S"]
            
            action_output = network_array_to_image(Q_values,node_names_list=action_node_names, layer_description="Action-Output", img_width=img_1D_size)

            motor_actions = network_array_to_image(Q_values[0][:len(motor_action_node_names)],node_names_list=motor_action_node_names, layer_description="Motor Actions", img_width=img_1D_size)
            verbal_actions = network_array_to_image(Q_values[0][len(motor_action_node_names):],node_names_list=verbal_action_node_names, layer_description="Verbal Actions", img_width=img_1D_size)

            action_output = get_concat_h(motor_actions, verbal_actions, distance = 50)
            
            ### set boundaries around 2d images

            #visual_input = ImageOps.expand(visual_input, border=border_width, fill = border_color)
            #cell_img = ImageOps.expand(cell_img, border=border_width, fill = border_color)
            #hidden_img = ImageOps.expand(hidden_img, border=border_width, fill = border_color)
            
            
            ######################
            ## Put images together
            #####################
            #input_img = get_concat_h(visual_input, task_image, distance = 50)
            ConvLSTM_imgs = get_concat_v(cell_img, hidden_img, distance = 50)
            visual_img = get_concat_v(visual_input, ConvLSTM_imgs, distance = 50)
            visual_img = get_concat_v(visual_img, vis_representation_layer_img_small, distance = 50)

            language_input_img = get_concat_h(task_image, verbal_actions, distance = 50)
            LSTM_imgs = get_concat_h(cell_lang_img, hidden_lang_img, distance = 50)
            auditory_img = get_concat_v(language_input_img, LSTM_imgs, distance = 50)
            
            #both_LSTM_imgs = get_concat_h(ConvLSTM_imgs, LSTM_imgs, distance = 50)
            
            whole_img = get_concat_h(visual_img, auditory_img, distance = 50)
            #whole_img = get_concat_v(whole_img, vis_representation_layer_img, distance = 50)
                        
            vis_lang_representation = get_concat_h(vis_representation_layer_img, hidden_lang_img, distance = 50)
            vis_lang_task_representation = get_concat_h(vis_lang_representation, task_image, distance = 50)
            
            whole_img = get_concat_v(whole_img, vis_lang_task_representation, distance = 50) 

            emtpy_img = Image.new('RGB', (motor_actions.width, output_lang_repr_img.height), color='white')
            output_lang_repr_img_extended = get_concat_h(emtpy_img, output_lang_repr_img_small, distance = 50)  

            whole_img = get_concat_v(whole_img, output_lang_repr_img_extended, distance = 50)            
            whole_img = get_concat_v(whole_img, action_output, distance = 50)


            big_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 40)
            big_description_img = Image.new('RGB', (whole_img.width, visual_input.width//4), color='white')
            draw = ImageDraw.Draw(big_description_img)
            wordy = node_names_list[env.task_n] + " " + node_names_list[10 + env.quant_n] + " "+ node_names_list[5 + env.object_n]  
            w, h = draw.textsize(wordy, font=big_font)
            W = big_description_img.width
            H = big_description_img.height
            draw.text(((W-w)/2,(H-h)/2),wordy,(0,0,0), font=big_font)
            whole_img = get_concat_v(big_description_img, whole_img, distance = 0)

            network_image_list.append(whole_img)
            
            
            
            #######################
            ## Display
            #####################
            update_display(whole_img, display_id = "game")
            time.sleep(delay_time*2)
            
           
            
            
             


          if(env.ended and display_==True):
              print("                               ")
              print("\\------------------------------/")
              print(" \\----------------------------/")
              print("  Congrats - successful trial!")   
              print(" /----------------------------\\")
              print("/------------------------------\\")
      
      ########
      ## save network activity
      if(save_network_path is not None):
        pathy = save_network_path + "/" + readable_task[env.task]
        network_image_list[0].save(pathy, format='GIF', append_images=network_image_list[1:], save_all=True, duration=500, loop=0) 

      
      ### get variability
      f_is = []
      action_sequence_length = len(a_is)
      
      for i in range(n_actions):
        if(action_sequence_length>0):
           f_is.append(a_is.count(i)/action_sequence_length)
        else: 
           f_is.append(0)
      
      sum_of_squared_f_is = 0
      for i in range(len(f_is)):
        sum_of_squared_f_is += f_is[i]*f_is[i]
      
      variability = 1-np.sqrt(sum_of_squared_f_is)
      #print("Variability: ", variability)
      
      if(PATH is not None):

        text_path = PATH + ".txt"
        gif_path = PATH + ".gif"
        all_text_path = model.model_path + "actions.txt"
        all_text_path_html = model.model_path + "actions.html"
        
        #print("saving gifs and action sequence...")
        #print(text_path)
        #print(gif_path)
        
        #save_gif(env, gif_path)
        #save_action_sequence(env, text_path, model.episode)
        save_action_sequence(env, all_text_path, model.episode)
        save_action_sequence_to_html(env, all_text_path_html, model.episode)
              
def save_gif(env, PATH, display = False):
    images = []
    new_size = 840
    background = Image.new("RGB", (new_size,new_size))
    #display(new_im, display_id = "game")
    white = (255,255,255)

    for t in range(0, len(env.epoch) ):

            old_size = 800
            img = env.epoch[t]['dem_img'].resize( (old_size,old_size) , resample=0).convert('RGB')

            old_im = img
            new_im = background
            new_im.paste(old_im, ( int( (new_size-old_size)/2) ,
                          int( (new_size-old_size)/2) ))
            
            #images.append( new_im.resize( (400,400) ) )
            images.append( img )
            if(display):
                update_display(new_im, display_id = "game")
                time.sleep(0.3)
    images.append(Image.new("RGB", (new_size,new_size),white))
    images[0].save(PATH, format='GIF', append_images=images[1:], save_all=True, duration=500, loop=0)              

    
    
def save_action_sequence(env, PATH, episode):
    whole_text = ""
    if(env.n_squares==1):
      whole_text += "\n Episode: " + str(episode)
      whole_text += "\n   Task: " + str(readable_task[env.task]) + "\n"
    whole_text += "     n= " + str(env.n_squares) + ": "
    for a in range(len(env.epoch)-1):
        #print( [a_strings[i] for i in range(len(a_strings)) if env.epoch[a]['action'][0].tolist()[i] and i!=env.n_actions-1  and i!=env.n_motor_actions] )
        whole_text += "   " + str([env.a_strings[0][i] for i in range(len(env.a_strings[0])) if env.epoch[a]['action'][0].tolist()[i] and i!=env.n_actions-1  and i!=env.n_motor_actions]) + ","
    whole_text += "\n"
    
    fily = open(PATH,"a+")
    fily.write(whole_text)
    fily.close()

def save_action_sequence_to_html(env, PATH, episode):
    whole_text = ""
    if(env.n_squares==1):
      whole_text += "<br> Episode: " + str(episode)
      whole_text += "<br>   Task: " + str(readable_task[env.task]) + "<br>"
    whole_text += "     n= " + str(env.n_squares) + ": "
    for a in range(len(env.epoch)-1):
        #print( [a_strings[i] for i in range(len(a_strings)) if env.epoch[a]['action'][0].tolist()[i] and i!=env.n_actions-1  and i!=env.n_motor_actions] )
        text = "   " + str([env.a_strings[0][i] for i in range(len(env.a_strings[0])) if env.epoch[a]['action'][0].tolist()[i] and i!=env.n_actions-1  and i!=env.n_motor_actions]) + ","
        back_ground_color, text_color = color_of_string(text)
        whole_text += "<span style=\"background-color: " + back_ground_color + ";color: " + text_color + " \"> " + text + "</span>"
    whole_text += "<br>"
    
    fily = open(PATH,"a+")
    fily.write(whole_text)
    fily.close()
def color_of_string(action_string):
  number_strings = ["1","2","3","4","5","6","7","8","9"]
  back_ground_colory = "#FFFFFF"
  colory = "black"

  if("Dr" in action_string):
    colory = "blue"

  if("P" in action_string):
    colory = "blue"

  if("E" in action_string):
    back_ground_colory = "orange"

  if any(ext in number_strings for ext in action_string):
     back_ground_colory = "yellow"

  return back_ground_colory, colory


def get_concat_h(im1, im2, distance = 50, colory='white'):
    dst = Image.new('RGB', (im1.width + im2.width + distance, im1.height), color=colory)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + distance, 0))
    return dst

def get_concat_v(im1, im2, distance=50, colory='white'):
    dst = Image.new('RGB', (max(im1.width,im2.width), im1.height + im2.height + distance), color=colory)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + distance))
    return dst   
 
def get_concat_h_embedded(im1, im2, distance = 50, colory='white'):
    dst = Image.new('RGB', (im1.width + im2.width + 3*distance, im1.height + 2*distance), color=colory)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + distance, 0))
    return dst

def get_concat_v_embedded(im1, im2, distance=50, colory='white'):
    dst = Image.new('RGB', (max(im1.width,im2.width) + 2*distance, im1.height + im2.height + 3*distance), color=colory)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + distance))
    return dst  



def network_array_to_image(array_,node_names_list=None, layer_description=None, img_width=45):
      
      array_ = array_.detach().numpy()
      array_size = array_.size

      array_ = array_.reshape(array_size)

      '''
      for i in range(task_vector.size):
        if(i%2==0):
          task_vector[i] = 0.5
      '''
      double_array_ = np.stack( (array_,array_))

      img_width = array_size*img_width
      img_height = img_width//array_size
      img = Image.fromarray(double_array_*255).resize( (img_width,img_height), resample=0).convert('RGB') 

      boundary_color = (0,0,255)
      boundary_width = 4

      drawy = ImageDraw.Draw(img)
      for i in range(array_size):
        drawy.line((i*img_height - boundary_width//2 ,0, i*img_height- boundary_width//2,img_height), fill=boundary_color, width = boundary_width )

      border_width = 5
      img = ImageOps.expand(img, border=border_width, fill = boundary_color)

      text_img = Image.new('RGB', (img.width, 3*img.height), color='white')

      task_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 20)
      node_font = ImageFont.truetype("/content/drive/My Drive/Embodied_counting/src/Arial.ttf", 12)


      #node_names_list = ["Touch", "Recite", "Count", "Give","Nothing","1", "2","3", "4","5", "6","7", "8","9", "ALL","Objects", "Events","-", "-", "-"]
      if(node_names_list is not None):
          for w_i in range(len(node_names_list)):

            text_img_i = Image.new('RGB', (img.height, img.height), color='white')
            draw = ImageDraw.Draw(text_img_i)
            wordy = node_names_list[w_i]
            draw.text((0, 0),wordy,(0,0,0), font=node_font)
            if(len(wordy)>2):

              text_img_i = Image.new('RGB', (2*img.height, img.height), color='white')
              draw = ImageDraw.Draw(text_img_i)
              wordy = node_names_list[w_i]
              draw.text((0, 0),wordy,(0,0,0), font=node_font)

              text_img_i = text_img_i.rotate(90, expand = 1)
              text_img.paste(text_img_i, (border_width + w_i*img_height + img_height//2 - 2*len(wordy) + 5, 1*img_height))
            else:
              text_img.paste(text_img_i, (border_width + w_i*img_height + img_height//2 - 3*len(wordy), 35 + 2*img_height))



      description_img = Image.new('RGB', (img.width, 2*img.height), color='white')
      draw_description = ImageDraw.Draw(description_img)
      if(layer_description is not None):
          wordy = layer_description

          draw_description.text((img_width//2 - 3*len(wordy), 0),wordy,(0,0,0),font=task_font)

      array_img = get_concat_v(text_img, img, distance=0)
      array_img = get_concat_v(array_img, description_img, distance=0)

      return array_img











