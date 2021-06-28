from copy import copy

from environment.task_utils import *
from environment.env_sub_classes import Square, Hand, Pos


import numpy as np
import random



print("Importing counting tasks...!!!")

class CountingTask():
  def __init__(self):
    self.vari = 0

  def initialize_env(self, env):
      initialize_hand(env, mode='random')
      initialize_squares(env, mode='random')
      for i in range(env.n_squares):
          env.aimed_count_list.append(str(i+1) )
          env.aimed_given_square_id_list.append(str(i+1) )
  
  def solve_task(self):
  	raise NotImplementedError

  def initialize_task_vector(self, env):
    pass

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self):
  	raise NotImplementedError

  def observation_change(self, env):
    pass
    
###########################
### How-Many
###########################
class HowMany(CountingTask):
  def __init__(self):
    self.vari=0
  
  def solve_task(self, env):
    done = False
    n = env.n_squares
    env.triple_update("touch", False, str(n), True)
  
  def initialize_task_vector(self, env):
    env.task_n = 0
    env.quant_n = 0
    env.object_n = 0
    env.hand_is_visible = False
    env.max_time = 2

  def update_variables(self, env):
    if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
      env.counted_word_list.append(env.action)
   
  def check_solved_condition(self, env):
    if(len(env.counted_word_list)>0):
      if(env.counted_word_list[0]==str(env.n_squares) ):
        env.ended = True



###########################
### More
###########################
class More(CountingTask):
  def __init__(self):
    self.vari=0
  
  def solve_task(self, env):
    done = False
    n = max(env.n_squares, env.n_squares_2)
    env.triple_update("touch", False, str(n), True)
  
  def initialize_task_vector(self, env):
    env.task_n = 4
    env.quant_n = env.n_squares
    env.quant_n_2 = env.n_squares_2
    env.object_n = 2
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = 2

  def update_variables(self, env):
    if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
      env.counted_word_list.append(env.action)
   
  def check_solved_condition(self, env):
    if(len(env.counted_word_list)>0):
      if(env.counted_word_list[0]==str( max(env.n_squares, env.n_squares_2) ) ):
        env.ended = True


###########################
### Successor
###########################
class Successor(CountingTask):
  def __init__(self, env):
    self.vari=0
    self.vari=0
    self.aimed_count_list_for_all_numbers = [create_successor(env, n) for n in range(1, env.n_squares_max+20)]

  
  def solve_task(self, env):
    successor_of_n(env)
  
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='empty')

    env_aimed_count_list_int = create_successor(env)
    env.aimed_count_list = [str(n_i) for n_i in env_aimed_count_list_int]
  
  def initialize_task_vector(self, env):
    env.task_n = 3
    env.quant_n = env.n_squares
    env.object_n = 2
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = 12

  def update_variables(self, env):
    if(env.IsSayNumberWord):
      env.counted_word_list.append(env.action)
   
  def check_solved_condition(self, env):
    if(env.counted_word_list==env.aimed_count_list):
        env.right_number_order = True
    if(env.counted_word_list==env.aimed_count_list and env.action=="stop" ): 
        env.ended = True

    # check if solved for another number
    for n_i in range(len(self.aimed_count_list_for_all_numbers)):
      if(env.counted_word_list==self.aimed_count_list_for_all_numbers[n_i] and env.action=="stop" ):
          env.solved_for_n = n_i+1


###########################
### Recite-N
###########################
class ReciteN(CountingTask):
  def __init__(self, env):
    self.vari=0
    self.aimed_count_list_for_all_numbers = [create_counting_sequence(env, n) for n in range(1, env.n_squares_max+20)]

  
  def solve_task(self, env):
  	recite_n(env)

  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='empty')

    env_aimed_count_list_int = create_counting_sequence(env)
    env.aimed_count_list = [str(n_i) for n_i in env_aimed_count_list_int]
  
  def initialize_task_vector(self, env):
    env.task_n = 1
    env.object_n = 2
    env.quant_n = env.n_squares
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = env.n_squares*2 + 4

  def update_variables(self, env):
    if(env.IsSayNumberWord):
      env.counted_word_list.append(env.action)
   
  def check_solved_condition(self, env):
    if(env.action == "stop"):
        if(env.counted_word_list!=env.aimed_count_list): 
            env.stopped_early = True
    if(env.counted_word_list==env.aimed_count_list):
        env.right_number_order = True
    if(env.counted_word_list==env.aimed_count_list and env.action=="stop" and env.stopped_early == False): 
        env.ended = True

    # check if solved for another number
    if(env.action=="stop" and not env.stopped_earlier_already):
      for n_i in range(len(self.aimed_count_list_for_all_numbers)):
        if(env.counted_word_list==self.aimed_count_list_for_all_numbers[n_i]):
          env.solved_for_n = n_i+1
    if(env.action=="stop"):
      env.stopped_earlier_already = True

    

###########################
### Count All Events
###########################
class CountAllEvents(CountingTask):
  def __init__(self):
    self.vari=0
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    env.squares = []
    pos = env.pseudo_pos #Pos(3, 3)
    pos.x = 3
    pos.y = 3
    new_square = env.pseudo_square  
    new_square.pos = pos
    env.squares.append(new_square)
    env.event_there = False
    env.square_is_visible = False
  
  def solve_task(self, env):
  	count_all_events(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 1
    env.object_n = 1
    env.quant_n = 0
    env.hand_is_visible = False
    env.max_time = env.n_squares*8
    for i in range(env.n_squares):
        env.aimed_count_list.append(str(i+1) )
        env.aimed_given_square_id_list.append(str(i+1) )

  def update_variables(self, env):
  	update_for_events(env)
   
  def check_solved_condition(self, env):
      if(env.one_step_after_event == True):
            if(env.count_action==False):
              env.missed_count = True
      if(env.count_action==True):
        if(env.one_step_after_event == False):
            env.counted_none_object = True
        
        if(env.missed_count == False and env.counted_none_object == False):
            env.one_to_one_correspondence = True
        else:
            env.one_to_one_correspondence = False
          
                
        if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
            if(env.one_step_after_event == True):
                env.counted_word_list.append(env.action)
                
        right_count_sequence = False
        if(env.counted_word_list==env.aimed_count_list):
            right_count_sequence = True
            env.right_number_order = True
          
        if(right_count_sequence):
          env.ended = True    


###########################
### Count All Objects
###########################
class CountAllObjects(CountingTask):
  def __init__(self):
    self.vari=0
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='random')
    for i in range(env.n_squares):
        env.aimed_count_list.append(str(i+1) )
        env.aimed_given_square_id_list.append(str(i+1) )
  
  def solve_task(self, env):
  	count_all_objects(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 1
    env.object_n = 0
    env.quant_n = 0
    env.max_time = env.n_squares*7

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self, env):
      if(env.IsTouch==True and Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
          env.counted_square_list.append(env.touched_square)
          env.counted_word_list.append(env.action)

      counted_every_square_exactly_once = True
      for i in range(len(env.squares)):
          if(env.counted_square_list.count(i+1) !=1  ):
              counted_every_square_exactly_once = False
              
      ## Check for one-to-one correspondence
      all_squares_touched_exactly_once = True
      for square in env.squares:
          if(square.touched_count != 1):
              all_squares_touched_exactly_once = False
      if(all_squares_touched_exactly_once):
        env.one_to_one_correspondence = True
      
      if(counted_every_square_exactly_once == True and env.counted_none_object==False):
        env.one_to_one_correspondence = True
      
      right_count_sequence = False
      if(env.counted_word_list==env.aimed_count_list):
          right_count_sequence = True
          env.right_number_order = True
        
      if(counted_every_square_exactly_once and right_count_sequence):
        env.ended = True



###########################
### Give-N
###########################
class GiveN(CountingTask):
  def __init__(self):
    self.vari=0
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    env.squares = []
    pos = env.pseudo_pos #Pos(3, 3)
    pos.x = 0
    pos.y = 0
    new_square = env.pseudo_square  
    new_square.pos = pos
    new_square.id_ = 1
    env.squares.append(new_square)
    env.obj_source = "infinite_squares"
    for i in range(env.n_squares):
        env.aimed_count_list.append(str(i+1) )
        env.aimed_given_square_id_list.append(str(i+1) )

  def solve_task(self, env):
  	give_n(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 2
    env.object_n = 0
    env.quant_n = env.n_squares  
    env.max_time = env.n_squares*9 + 6

  def update_variables(self, env):
    
    if(env.action=="release"):
      #print("env.IsGrab: ", env.IsGrab)
      if(env.IsGrab==True and env.hand.pos.y == env.img_size-1):
          env.given_square_id_list.append(str(env.grabed_square) )
          env.steps_after_given_square = 0
          self.IsGrab = False 

    if( int(env.action == Action_inv[env.action]) < 4 and env.pick_from_00 == True and env.obj_source == "infinite_squares"):
      pos = Pos(0, 0)
      env.give_n_current_squares += 1
      new_square = Square(pos, env.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
      env.squares.append(new_square)
      env.pick_from_00 = False

    if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
      env.counted_word_list.append(env.action)
    
    if( env.pick_from_00==True and env.action == "release"):
      env.pick_from_00 = False 
   
  def check_solved_condition(self, env):
      if(env.action == "stop"):
          if(env.given_square_id_list != env.aimed_given_square_id_list or env.counted_word_list!=env.aimed_count_list): 
              env.stopped_early = True
      if(env.counted_word_list==env.aimed_count_list):
          env.right_number_order = True
      if(env.given_square_id_list == env.aimed_given_square_id_list and env.counted_word_list==env.aimed_count_list and env.action=="stop" and env.stopped_early == False): 
          env.ended = True

      # Analyse error
      if(env.steps_after_given_square == 1):
          if(env.count_action==False):
              env.missed_count = True
      if(env.count_action==True):
          if(env.steps_after_given_square != 1):
              env.counted_none_object = True
        
      if(env.missed_count == False and env.counted_none_object == False):
          env.one_to_one_correspondence = True
      else:
          env.one_to_one_correspondence = False
        
      if(env.second_action == True): 
          env.steps_after_given_square += 1

  def observation_change(self, env):
      pass
 




###########################
### Give-N Extended: distributed squares + high numbers
###########################
class GiveNExtended(CountingTask):
  def __init__(self, env):
    self.vari=0
    self.aimed_count_list_for_all_numbers = [create_counting_sequence(env, n) for n in range(1, env.n_squares_max+20)]
    self.aimed_given_square_id_list_for_all_numbers = []

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    env.obj_source = "random" if IsDistributedSquares else "infinite_squares"

    for n_squares in range(1, env.n_squares_max+20):
      aimed_given_square_id_list = [str(n) for n in range(1,n_squares+1)]
      self.aimed_given_square_id_list_for_all_numbers.append(aimed_given_square_id_list )
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='give_n')


    env_aimed_count_list_int = create_counting_sequence(env)
    env.aimed_count_list = [str(n_i) for n_i in env_aimed_count_list_int]
    for i in range(env.n_squares):
        #env.aimed_count_list.append(str(i+1) )
        env.aimed_given_square_id_list.append(str(i+1) )

  def solve_task(self, env):
  	give_n(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 2
    env.object_n = 0
    env.quant_n = env.n_squares  
    env.max_time = env.n_squares*(env.img_size*4) + 10

  def update_variables(self, env):
    
    if(env.action=="release"):
      #print("env.IsGrab: ", env.IsGrab)
      if(env.hand.pos.y == env.img_size-1):
          env.given_square_id_list.append(str(env.grabed_square) )
          env.steps_after_given_square = 0
          #self.IsGrab = False 

    if( int(env.action == env.Action_Dict_inv[env.action]) < 4 and env.pick_from_00 == True and env.obj_source == "infinite_squares"):
      pos = Pos(0, 0)
      env.give_n_current_squares += 1
      new_square = Square(pos, env.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
      env.squares.append(new_square)
      env.pick_from_00 = False

    if(env.IsSayNumberWord):
      env.counted_word_list.append(env.action)
    
    if( env.pick_from_00==True and env.action == "release"):
      env.pick_from_00 = False 
   
  def check_solved_condition(self, env):
      if(env.action == "stop"):
          if(len(env.given_square_id_list) != len(env.aimed_given_square_id_list) or env.counted_word_list!=env.aimed_count_list): 
              env.stopped_early = True
      if(env.counted_word_list==env.aimed_count_list):
          env.right_number_order = True
      if(len(env.given_square_id_list) == len(env.aimed_given_square_id_list)  and not checkIfDuplicates(env.given_square_id_list) and env.counted_word_list==env.aimed_count_list and env.action=="stop" and env.stopped_early == False): 
          env.ended = True

      if(env.counted_word_list == env.aimed_count_list[:len(env.counted_word_list)]):
          env.did_wrong_action = False
      else:
          env.did_wrong_action = True

      # check if solved for another number
      if(env.action=="stop" and not env.stopped_earlier_already):
        for n_i in range(len(self.aimed_count_list_for_all_numbers)):
          if(len(env.given_square_id_list) == len(self.aimed_given_square_id_list_for_all_numbers[n_i])  and not checkIfDuplicates(self.aimed_given_square_id_list_for_all_numbers[n_i]) and env.counted_word_list==self.aimed_count_list_for_all_numbers[n_i]):
            env.solved_for_n = n_i+1
      if(env.action=="stop"):
        env.stopped_earlier_already = True

      # Analyse error
      if(env.steps_after_given_square == 1):
          if(env.count_action==False):
              env.missed_count = True
      if(env.count_action==True):
          if(env.steps_after_given_square != 1):
              env.counted_none_object = True
        
      if(env.missed_count == False and env.counted_none_object == False):
          env.one_to_one_correspondence = True
      else:
          env.one_to_one_correspondence = False
        
      if(env.second_action == True): 
          env.steps_after_given_square += 1

  def observation_change(self, env):
      pass


###########################
### Give one more
###########################
class GiveOneMore(GiveNExtended):
  def __init__(self, env):
    self.vari=0
    self.aimed_count_list_for_all_numbers  = [create_successor(env, n) for n in range(1, env.n_squares_max+20)]
    self.aimed_given_square_id_list_for_all_numbers = []

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    env.obj_source = "random" if IsDistributedSquares else "infinite_squares"

    for n_squares in range(1, env.n_squares_max+20):
      aimed_given_square_id_list = ['1']
      self.aimed_given_square_id_list_for_all_numbers.append(aimed_given_square_id_list )

  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='give_n')

    env_aimed_count_list_int = create_successor(env)
    env.aimed_count_list = [str(n_i) for n_i in env_aimed_count_list_int]
    #self.aimed_count_list_for_all_numbers = [create_successor(env, n) for n in range(1, env.n_squares_max+20)]
    env.aimed_given_square_id_list = ['1']

  def solve_task(self, env):
    give_one_more(env)  


  def initialize_task_vector(self, env):
    env.task_n = 3
    env.quant_n = env.n_squares
    env.object_n = 2
    env.hand_is_visible = True
    env.square_is_visible = True
    env.max_time = 22


###########################
### Give-N Extended wo numbers
###########################
class GiveNExtendedWoNumbers(CountingTask):
  def __init__(self, env):
    self.vari=0
    self.aimed_count_list_for_all_numbers = [create_counting_sequence(env, n) for n in range(1, env.n_squares_max+20)]
    self.aimed_given_square_id_list_for_all_numbers = []

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    env.obj_source = "random" if IsDistributedSquares else "infinite_squares"

    for n_squares in range(1, env.n_squares_max+20):
      aimed_given_square_id_list = [str(n) for n in range(1,n_squares+1)]
      self.aimed_given_square_id_list_for_all_numbers.append(aimed_given_square_id_list )

  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    initialize_squares(env, mode='give_n')

    for i in range(env.n_squares):
        #env.aimed_count_list.append(str(i+1) )
        env.aimed_given_square_id_list.append(str(i+1) )

  def solve_task(self, env):
  	give_n_wo_numbers(env)
  	
  
  def initialize_task_vector(self, env):
    env.task_n = 2
    env.object_n = 0
    env.quant_n = env.n_squares  
    env.max_time = env.n_squares*(env.img_size*4) + 10

  def update_variables(self, env):
    
    if(env.action=="release"):
      #print("env.IsGrab: ", env.IsGrab)
      if(env.hand.pos.y == env.img_size-1):
          env.given_square_id_list.append(str(env.grabed_square) )
          env.steps_after_given_square = 0
          #self.IsGrab = False 

    if( int(env.action == env.Action_Dict_inv[env.action]) < 4 and env.pick_from_00 == True and env.obj_source == "infinite_squares"):
      pos = Pos(0, 0)
      env.give_n_current_squares += 1
      new_square = Square(pos, env.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
      env.squares.append(new_square)
      env.pick_from_00 = False

    
    if( env.pick_from_00==True and env.action == "release"):
      env.pick_from_00 = False 
   
  def check_solved_condition(self, env):
      if(env.action == "stop"):
          if(len(env.given_square_id_list) != len(env.aimed_given_square_id_list)): 
              env.stopped_early = True
      if(len(env.given_square_id_list) == len(env.aimed_given_square_id_list) and not checkIfDuplicates(env.given_square_id_list) and env.action=="stop" and env.stopped_early == False): 
          env.ended = True

      # check if solved for another number
      if(env.action=="stop" and not env.stopped_earlier_already):
        for n_i in range(len(self.aimed_count_list_for_all_numbers)):
          if(len(env.given_square_id_list) == len(self.aimed_given_square_id_list_for_all_numbers[n_i]) and not checkIfDuplicates(self.aimed_given_square_id_list_for_all_numbers[n_i])):
            env.solved_for_n = n_i+1
      if(env.action=="stop"):
        env.stopped_earlier_already = True

      # Analyse error
      if(env.steps_after_given_square == 1):
          if(env.count_action==False):
              env.missed_count = True
      if(env.count_action==True):
          if(env.steps_after_given_square != 1):
              env.counted_none_object = True
        
      if(env.missed_count == False and env.counted_none_object == False):
          env.one_to_one_correspondence = True
      else:
          env.one_to_one_correspondence = False
        
      if(env.second_action == True): 
          env.steps_after_given_square += 1

  def observation_change(self, env):
      pass








###########################
### TOUCH ALL OBJECTS
###########################
class TouchAllObjects(CountingTask):
  def __init__(self):
    self.vari=0
  
  def solve_task(self, env):
  	touch_all_objects(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 0
    env.object_n = 0
    env.quant_n = 0
    env.max_time = env.n_squares*10

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self, env):
      all_squares_touched_exactly_once = True
      for square in env.squares:
          if(square.touched_count != 1):
              all_squares_touched_exactly_once = False
      if(all_squares_touched_exactly_once and env.counted_none_object==False):
        env.one_to_one_correspondence = True
      if(all_squares_touched_exactly_once):
        env.ended = True



###########################
### GIVE AND TAKE
###########################
class GiveAndTake(CountingTask):
  def __init__(self):
    self.vari=0
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    env.squares = []
    pos = env.pseudo_pos #Pos(3, 3)
    pos.x = 0
    pos.y = 0
    new_square = env.pseudo_square  
    new_square.pos = pos
    new_square.id_ = 1
    env.squares.append(new_square)
    env.obj_source = "infinite_squares"
    env.pick_from_03 = False

    for i in range(env.n_squares):
        env.aimed_count_list.append(str(i+1) )
    for i in range(env.n_squares):
        env.aimed_count_list.append(str(i+1) )
    for i in range(env.n_squares):
        env.aimed_given_square_id_list.append(str(env.n_squares-i) )


  def solve_task(self, env):
  	give_and_take(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 4
    env.object_n = 0
    env.quant_n = env.n_squares  
    env.max_time = env.n_squares*30 + 6
    

  def update_variables(self, env):

    # Give-N part: replace square when moving from spot with square in hand
    if( int(env.action == Action_inv[env.action]) < 4 and env.pick_from_00 == True and env.obj_source == "infinite_squares"):
        pos = Pos(0, 0)
        env.give_n_current_squares += 1
        new_square = Square(pos, env.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square)
        env.pick_from_00 = False

        if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
          env.counted_word_list.append(env.action)
        
        if( env.pick_from_00==True and env.action == "release"):
          env.pick_from_00 = False


    if(env.action=="pick"):
      env.action_onehot = np.array([int(i == 4) for i in range(env.n_actions)])
      if(env.obj_belong_fct[env.hand.pos.x, env.hand.pos.y] != 0):
        if(env.hand.pos.x == 0 and env.hand.pos.y==3):
          env.pick_from_03 = True          
        if(env.picked_once_already == False):            
          env.picked = True
        env.picked_once_already = True

    if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
      env.counted_word_list.append(env.action)

    # Check if all squares taken to final spot
    if(env.action == "release"and env.pick_from_03 == True):
        if(env.hand.pos.x == 3 and env.hand.pos.y == 0):
            env.given_square_id_list.append(str(env.grabed_square) )
            env.pick_from_03 = False
       
  def check_solved_condition(self, env):
      if(env.action == "stop"):
          if(env.given_square_id_list != env.aimed_given_square_id_list or env.counted_word_list!=env.aimed_count_list): 
              env.stopped_early = True
      if(env.counted_word_list==env.aimed_count_list):
          env.right_number_order = True
      if(env.given_square_id_list == env.aimed_given_square_id_list and env.counted_word_list==env.aimed_count_list and env.action=="stop" and env.stopped_early == False): 
          env.ended = True

  def observation_change(self, env):
      env.observation[0, env.img_size-1] = 150


###########################
### GIVE AND TAKE WO Counting
###########################
class GiveAndTakeWOCounting(GiveAndTake):
  def __init__(self):
    self.vari=0    

  def solve_task(self, env):
  	give_and_take_wo_numbers(env)


  def check_solved_condition(self, env):
      if(env.action == "stop"):
          if(env.given_square_id_list != env.aimed_given_square_id_list): 
              env.stopped_early = True
      if(env.given_square_id_list == env.aimed_given_square_id_list and env.action=="stop" and env.stopped_early == False): 
          env.ended = True

###########################
### GIVE GIVE AND TAKE
###########################
class GiveGiveAndTake(CountingTask):
  def __init__(self):
    self.vari=0
    
  def initialize_env(self, env):
    initialize_hand(env, mode='random')
    env.squares = []
    pos = env.pseudo_pos #Pos(3, 3)
    pos.x = 0
    pos.y = 0
    new_square = env.pseudo_square  
    new_square.pos = pos
    new_square.id_ = 1
    env.squares.append(new_square)
    env.obj_source = "infinite_squares"
  
  def solve_task(self, env):
  	give_give_and_take(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 4
    env.object_n = 0
    env.quant_n = env.n_squares  
    env.max_time = env.n_squares*30 + 6

  def update_variables(self, env):
    if( int(env.action == Action_inv[env.action]) < 4 and env.pick_from_00 == True and env.obj_source == "infinite_squares"):
      pos = Pos(0, 0)
      env.give_n_current_squares += 1
      new_square = Square(pos, env.give_n_current_squares, 0 )  #(data,) pos, id_, n_neighbours
      env.squares.append(new_square)
      env.pick_from_00 = False
    
    if( env.pick_from_00==True and env.action == "release"):
      env.pick_from_00 = False 
   
  def check_solved_condition(self, env):
  	check_counted_word_list(env)

  def observation_change(self, env):
      env.observation[0, env.img_size-1] = 150


###########################
### Do NOTHING
###########################
class DoNothing(CountingTask):
  def __init__(self):
    self.vari=0
    
  
  def solve_task(self, env):
  	do_nothing(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 4
    env.object_n = 1
    env.quant_n = 0
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = env.n_squares*2
    env.did_nothing = True

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self, env):
      if(env.action != "stop"):
          env.did_nothing = False
      if(env.time>env.n_squares-2 and env.did_nothing):
          env.ended = True


###########################
### Count On
###########################
class CountOn(CountingTask):
  def __init__(self):
    self.vari=0
    
  
  def solve_task(self, env):
  	count_on(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 4
    env.object_n = env.add_n
    env.quant_n = env.n_squares
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = env.n_squares*2 + env.add_n
    env.aimed_count_list = []
    for i in range(env.add_n):
      env.aimed_count_list.append(str(env.n_squares+1+i) )

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self, env):
  	check_counted_word_list(env)

###########################
### Recite-N Inverse
###########################
class ReciteNInverse(CountingTask):
  def __init__(self):
    self.vari=0
    
  
  def solve_task(self, env):
  	recite_n_inverse(env)
  
  def initialize_task_vector(self, env):
    env.task_n = 5
    env.object_n = 0
    env.quant_n = env.n_squares
    env.hand_is_visible = False
    env.square_is_visible = False
    env.max_time = env.n_squares*2
    for i in range(env.n_squares):
        env.aimed_count_list.append(str(env.n_squares-i) )

  def update_variables(self, env):
  	pass
   
  def check_solved_condition(self):
  	raise NotImplementedError













def touch_all_objects(env):
  
  isFoundAny = True
   
  while(isFoundAny):
    n, isFoundAny = find_next_object(env)
    if(isFoundAny):
        
        move_to_square(env, n)
        env.triple_update("touch", True, "1", False)


              
def count_all_objects(env):
  
  isFoundAny = True
  n_sofar=0
  
  while(isFoundAny):
    n, isFoundAny = find_next_object(env)
    n_sofar += 1
    if(isFoundAny):
        
        move_to_square(env, n)       
        env.triple_update("touch", True, str(n_sofar), True)
    
    

def move_all_squares_from_source_to_target(env):
    done = False  
    while(not done):
        done = move_square_from_source_to_target(env)


def give_n(env):
  
    done = False
    n = env.n_squares
    given_squares_so_far = 0

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        if(not IsDistributedSquares):
          done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,env.img_size-1])
          
        else:
              except_pos = Pos(0, env.img_size-1)
              n_found, isFoundAny = find_next_object(env, except_pos = except_pos)
              next_square_pos = [env.squares[n_found].pos.x, env.squares[n_found].pos.y]
              done = move_square_from_to(env, from_pos=next_square_pos, to_pos=[0,env.img_size-1])

        if(env.IsDecimal):
          n_string = int2base(given_squares_so_far, base=env.base)
          if(env.leading_zero):
            n_string = "{:02d}".format(int(n_string))
          for s_i in n_string:
            env.triple_update("touch", False, s_i, True)
        if(not env.IsDecimal):
            env.triple_update("touch", False, str(given_squares_so_far), True)
    
    env.update("stop") 
    env.update("stop") 


def give_one_more(env):
  
    done = False
    n = env.n_squares
    n_successor = n + 1
    #given_squares_so_far = 0

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    
    #given_squares_so_far += 1
    if(not IsDistributedSquares):
      done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,env.img_size-1])
      
    else:
          except_pos = Pos(0, env.img_size-1)
          n_found, isFoundAny = find_next_object(env, except_pos = except_pos)
          next_square_pos = [env.squares[n_found].pos.x, env.squares[n_found].pos.y]
          done = move_square_from_to(env, from_pos=next_square_pos, to_pos=[0,env.img_size-1])

    if(env.IsDecimal):
      n_string = int2base(n_successor, base=env.base)
      if(env.leading_zero):
        n_string = "{:02d}".format(int(n_string))
      for s_i in n_string:
        env.triple_update("touch", False, s_i, True)
    if(not env.IsDecimal):
        env.triple_update("touch", False, str(n_successor), True)
    
    env.update("stop") 
    env.update("stop") 

 

def give_n_wo_numbers(env):
  
    done = False
    n = env.n_squares
    given_squares_so_far = 0

    IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        if(not IsDistributedSquares):
          done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,env.img_size-1])
          
        else:
              except_pos = Pos(0, env.img_size-1)
              n_found, isFoundAny = find_next_object(env, except_pos = except_pos)
              next_square_pos = [env.squares[n_found].pos.x, env.squares[n_found].pos.y]
              done = move_square_from_to(env, from_pos=next_square_pos, to_pos=[0,env.img_size-1])
    
    env.update("stop") 
    env.update("stop") 



def give_and_take(env):
  
    done = False
    n = env.n_squares
    given_squares_so_far = 0
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,3])
        env.triple_update("touch", False, str(given_squares_so_far), True)

    while(given_squares_so_far != 0):        
        given_squares_so_far -= 1
        done = move_square_from_to(env, from_pos=[0,3], to_pos=[3,0])
        env.triple_update("touch", False, str(n-given_squares_so_far), True)
        
     
    env.update("stop") 
    env.update("stop")

def give_and_take_wo_numbers(env):
  
    done = False
    n = env.n_squares
    given_squares_so_far = 0
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,3])

    while(given_squares_so_far != 0):        
        given_squares_so_far -= 1
        done = move_square_from_to(env, from_pos=[0,3], to_pos=[3,0])
        
     
    env.update("stop") 
    env.update("stop")

def give_give_and_take(env):
  
    done = False
    n = env.n_squares
    n2 = env.n_squares_2
    given_squares_so_far = 0
    
    
    while(given_squares_so_far != n):
        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,3])
        env.triple_update("touch", False, str(given_squares_so_far), True)

    given_squares_so_far = 0
    while(given_squares_so_far != n2):
        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,3])
        env.triple_update("touch", False, str(given_squares_so_far), True)

    given_squares_so_far = 0
    while(given_squares_so_far != (n+n2)):        
        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,3], to_pos=[3,0])
        env.triple_update("touch", False, str(given_squares_so_far), True)
        
     
    env.update("stop") 
    env.update("stop")
    
    
def count_all_events(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):
        
        event_there = copy(env.event_there)
        
        if(event_there):
            n_sofar += 1
            ########
            if(env.IsDecimal):
              n_string = int2base(n_sofar, base=env.base)
              if(env.leading_zero):
                n_string = "{:02d}".format(int(n_string))
              for s_i in n_string:
                env.triple_update("touch", False, s_i, True)
            if(not env.IsDecimal):
                env.triple_update("touch", False, str(n_sofar), True)

            ######
        else: 
            env.triple_update("down", True, "stop", False) 

                                                        
              
def recite_n(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0

    env.triple_update("right", True, str(n_sofar), False)
    env.triple_update("right", True, str(n_sofar), False)

    while(n_sofar < n):      
        n_sofar += 1
        if(env.IsDecimal):
          n_string = int2base(n_sofar, base=env.base)
          if(env.leading_zero):
            n_string = "{:02d}".format(int(n_string))
          for s_i in n_string:
            env.triple_update("touch", False, s_i, True)
        if(not env.IsDecimal):
            env.triple_update("touch", False, str(n_sofar), True)
     
    env.update("stop")
    env.update("stop")  


def successor_of_n(env):
  
    done = False
    n = env.n_squares

    env.triple_update("touch", True, str(1), False)
    env.triple_update("touch", True, str(1), False)

    for n_str in env.aimed_count_list:
      env.triple_update("touch", False, str(n_str), True)  

    env.update("stop") 


#### Inverse!!!!
def recite_n_inverse(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):               
        env.triple_update("touch", False, str(n-n_sofar), True)
        n_sofar += 1       
    env.update("stop")  

    
def count_on(env):
  
    done = False
    init_n = env.n_squares + 1
    n_sofar = copy(init_n)
    add_n = env.add_n
    
    for i in range(add_n):              
        env.triple_update("touch", False, str(init_n + i), True)
        n_sofar += 1    
    env.update("stop")  

    
    
def do_nothing(env):  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):     
        n_sofar += 1
        env.triple_update("touch", False, "stop", True) 



def update_for_events(env):
  # COUNT ALL EVENTS      
  if(env.task=="count_all_events" and env.second_action == True):        
    env.last_event_count+=1
    
    env.last_event_count==1

    if(env.last_event_count>env.n_wait_steps):
        env.event_there = True  
        env.event_had_occured_already = True
        env.n_wait_steps = random.randint(1,env.max_n_wait_steps)
        env.last_event_count = 0
        env.square_is_visible = True

        env.squares = []
        pos = Pos(1, 1)
        new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square)

        pos = Pos(2, 1)
        new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square)

        pos = Pos(1, 2)
        new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square)

        pos = Pos(2, 2)
        new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square)

    else:
        env.square_is_visible = False
        env.squares = []
        pos = Pos(3, 3)
        new_square = Square(pos, 0, 0 )  #(data,) pos, id_, n_neighbours
        env.squares.append(new_square) 
        env.event_there = False
        
    env.since_last_event_count = env.last_event_count
    env.one_step_after_event = False
    if(env.last_event_count == 1 and env.event_had_occured_already == True):
        env.one_step_after_event = True



def check_counted_word_list(env):
    if(Action_inv[env.action]>env.n_motor_actions and Action_inv[env.action]<17):
        if(env.last_action_is_count == True):
            env.counted_word_list.append(env.action)
        else:
            env.last_action_is_count = False
            
    right_count_sequence = False
    if(env.counted_word_list==env.aimed_count_list):
        right_count_sequence = True
        env.right_number_order = True
      
    if(right_count_sequence and env.action=="stop"):
      env.ended = True

def check_sub_task_success(env):
      if(env.IsTripleAction == False):
        env.time += 1
        if(env.print_sub_tasks_after_steps == True):
          print("One-to-one correspondence: ", env.one_to_one_correspondence)
          print("Right number sequence: ", env.right_number_order)
      else:
        if(env.second_action == True):
          env.time += 1
          if(env.print_sub_tasks_after_steps == True):
            print("One-to-one correspondence: ", env.one_to_one_correspondence)
            print("Right number sequence: ", env.right_number_order)









def initialize_hand(env, mode='random'):
        # Hand
        pointer, pointer_mask = np.array([[255]]), np.array([[255]])
        pointer_grab, _ = np.array([[255]]), np.array([[255]])
        hand_pos_x = random.randint(0,env.img_size-1) 
        hand_pos_y = random.randint(0,env.img_size-1) 
        pos = Pos(hand_pos_x, hand_pos_y)
        hand = Hand(pointer,pointer_mask, pos)
        hand_grab = Hand(pointer_grab,pointer_mask, pos)
        env.hand = hand        
        env.hand_nongrab = copy(hand)
        env.hand_grab = copy(hand_grab)

def initialize_squares(env, mode='random', exclude_pos=None, forced_n = None):
        # Squares with random positions
        squares = [] #Create_N_Sqaures(self.n_squares, mode = self.mode, max_dist = self.max_dist, img_size = self.img_size)
        pos_list = []
        if(mode=='random' or mode =='left'):
          n_maxy = env.n_squares
          if(forced_n is not None):
            n_maxy = forced_n

          for n in range(n_maxy):
              
              pos_not_ok = True
              trialsy = 0
              while(pos_not_ok):
                  
                  rand_pixel_1 = random.randint(0,env.img_size-1) 
                  rand_pixel_2 = random.randint(0,env.img_size-1) 
                  if(mode=='left'):
                    rand_pixel_2 = random.randint(0,env.img_size//2-1) 
                  pos_array = np.array([rand_pixel_1, rand_pixel_2])
                  
                  if(any((pos_array == x).all() for x in pos_list)):
                      pos_not_ok = True
                  else:
                      pos_not_ok = False
                      pos_list.append(pos_array)
                      if(exclude_pos is not None):
                        if((pos_array == exclude_pos).all() ):
                          pos_not_ok = True

                  trialsy += 1
                  if(trialsy>1000):
                    print("Tried too many times to fit squares")
                                        
              pos = Pos(rand_pixel_1, rand_pixel_2)
              square_now = Square(pos, n+1, 0 )  #data, pos, id_, n_neighbours
              squares.append(square_now)
              env.squares = squares
        if(mode=='one_at_edge'): 
              pos = Pos(0, 0)
              square_now = Square(pos, 1, 0 )  #data, pos, id_, n_neighbours
              squares.append(square_now)
              env.squares = squares

        if(mode=='empty'):
          env.squares = []


        if(mode=='give_n'):
          IsDistributedSquares = env.params['distributed_squares'] if 'distributed_squares' in env.params else False 
          #print("in mode='give_n'")
          #print("IsDistributedSquares", IsDistributedSquares)
          if(not IsDistributedSquares):
              env.squares = []
              pos = env.pseudo_pos #Pos(3, 3)
              pos.x = 0
              pos.y = 0
              new_square = env.pseudo_square  
              new_square.pos = pos
              new_square.id_ = 1
              env.squares.append(new_square)
              env.obj_source = "infinite_squares"
          else:
              initialize_squares(env, mode='random', exclude_pos = np.array([0, env.img_size-1]), forced_n = env.n_squares + random.randint(0,4) )
        
   


def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True



readable_task = {
    "touch_all_objects": "Touch all objects",
    "count_all_objects": "Count all objects",
    "count_all_events": "Count all events",
    "give_n": "Give N",
    "give_n_extended_wo_numbers": "Give N w.o. Number Words",
    "give_n_extended": "Give N",
    "give_one_more": "Give One More",
    "recite_n": "Recite N",   
    "do_nothing": "Do nothing",
    "recite_n_inverse": "Recite N inverse",
    "count_on": "Count On",
    "successor": "Successor",
    "more": "More"
}