##################################
## Automatic solving algorithm
##################################


print("Loading Automatic Solving Algorithms..?")

import math

delay_time = 0.5

def find_next_object(env, source="all"):
  
    d_min = 1e8
    square_n = -1
    n = - 1
    
    isFoundAny = False
    
    for square in env.squares:
        
        square_criterion = True
        if(env.task=="give_n"):
            square_criterion = square.picked_already
        else:
            square_criterion = square.touched_already
        
        n += 1        
        criterion = True
        if(source == "left"):
            criterion = False
            if( (square.pos.y < int(env.img_size/2.0)) and (square_criterion==False) ):
                criterion = True
                isFoundAny = True

        if(source == "right"):
            criterion = False
            if( (square.pos.y == 3) and (square.pos.x == 0) and (square_criterion==False) ):
                criterion = True
                isFoundAny = True
           
        # if source=="all", find closest one
        if(source == "all"): 
          if(criterion):
              d_curr = (env.hand.pos.x - square.pos.x)**2 + (env.hand.pos.y - square.pos.y)**2
              if((d_curr < d_min) and (not square_criterion) ):
                  d_min = d_curr
                  square_n = n
                  isFoundAny = True

        if isinstance(source, list): 
            if( (square.pos.y == source[1]) and (square.pos.x == source[0]) ):
              square_n = n
              isFoundAny = True
        

                
    return square_n, isFoundAny
  



def move_to_square(env, n):
    
      reached = False
      
      squ_x = env.squares[n].pos.x
      squ_y = env.squares[n].pos.y
      
      hand_x = env.hand.pos.x 
      hand_y = env.hand.pos.y 


      d_x = hand_x - squ_x
      d_y = hand_y - squ_y
      
      while(not reached):

          action = "none"
          if(abs(d_x) > abs(d_y)):

              if(d_x > 0):
                  env.update("up")
                  action = "up"
              else:  
                  env.update("down")
                  action = "down"
          else:
          
              if(d_y > 0):
                  env.update("left")
                  action = "left"
              else:  
                  env.update("right")
                  action = "right"
   
          hand_x = env.hand.pos.x 
          hand_y = env.hand.pos.y
          
          d_x = hand_x - squ_x
          d_y = hand_y - squ_y

          

          if(hand_x==squ_x and hand_y==squ_y):
              reached = True
              
      
    
def move_to(env, to_pos):
    
      reached = False
      
      to_pos_x = to_pos[0]
      to_pos_y = to_pos[1]
      
      hand_x = env.hand.pos.x 
      hand_y = env.hand.pos.y 


      d_x = hand_x - to_pos_x
      d_y = hand_y - to_pos_y
      
      while(not reached):
          if(abs(d_x) > abs(d_y)):

              if(d_x > 0):
                  env.update("up")
              else:  
                  env.update("down")
          else:
          
              if(d_y > 0):
                  env.update("left")
              else:  
                  env.update("right")
  
          hand_x = env.hand.pos.x 
          hand_y = env.hand.pos.y

          d_x = hand_x - to_pos_x
          d_y = hand_y - to_pos_y
          
          if(hand_x==to_pos_x and hand_y==to_pos_y):
              reached = True



def move_to_target(env):
  
    reached = False
    while(not reached):
        
        env.update("right")
        if(env.hand.pos.y >= int(0.75*env.img_size)):           
            reached = True

def move_to_target_2(env):
  
    reached = False
    while(not reached):       
        env.update("down")
        if(env.hand.pos.x >= int(0.75*env.img_size)):
            reached = True

            
    reached = False
    while(not reached):       
        env.update("down")
        if(env.hand.pos.y == 0):
            reached = True
        env.update("left")
        if(env.hand.pos.y == 0 ):
            reached = True



def pick_next_object(env):
    
    n, isFoundAny = find_next_object(env)
    move_to_square(env, n)
    env.update("pick")


def move_square_from_source_to_target(env, n_sofar = 0):
    
    n, isFoundAny = find_next_object(env, source = "left" )
    
    if(isFoundAny):
        move_to_square(env, n)
        env.update("pick")
        move_to_target(env)
        
        if(env.task == "give_n"):      
          if(n_sofar==env.n_squares):
              env.triple_update("release", True, "stop", False)
          else:
              env.update("release")
        else: 
           env.update("release")          

    return (not isFoundAny)


def move_square_from_to(env, from_pos=[0,0], to_pos=[0,0]):
    
    n, isFoundAny = find_next_object(env, source = from_pos )
        
    if(isFoundAny):
        
        move_to_square(env, n)
        env.update("pick")
        move_to(env, to_pos)
        env.update("release")          

    return (not isFoundAny)



def touch_all_objects(env):
  
  isFoundAny = True
   
  while(isFoundAny):
    n, isFoundAny = find_next_object(env, source = "all" )
    if(isFoundAny):
        
        move_to_square(env, n)
        env.triple_update("touch", True, "1", False)



              
def count_all_objects(env):
  
  isFoundAny = True
  n_sofar=0
  
  while(isFoundAny):
    n, isFoundAny = find_next_object(env, source = "all" )
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
    
    while(given_squares_so_far != n):

        given_squares_so_far += 1
        done = move_square_from_to(env, from_pos=[0,0], to_pos=[0,env.img_size-1])
        for s_i in str(given_squares_so_far):
          env.triple_update("touch", False, s_i, True)
    
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


    
    
def count_all_events(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):
        
        event_there = copy(env.event_there)
        
        if(event_there):
            n_sofar += 1
            env.triple_update("touch", False, str(n_sofar), True)

        else: 
            env.triple_update("down", True, "stop", False) 

                                                        
              
def recite_n(env):
  
    done = False
    n = env.n_squares
    n_sofar = 0
    
    while(n_sofar < n):      
        n_sofar += 1
        env.triple_update("touch", False, str(n_sofar), True)
     
    env.update("stop")  



#### Inverse!!!!
def recite_n_(env):
  
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

        




