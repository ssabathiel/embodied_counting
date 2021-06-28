print("Loading Automatic Solving Algorithms..?")

import math
import string


delay_time = 0.5

def find_next_object(env, except_pos = None):
  
    d_min = 1e8
    square_n = -1
    n = - 1
    
    isFoundAny = False
    
    for square in env.squares:
        n+=1
        square_criterion = True
        if(except_pos is not None):
          if((square.pos.x == except_pos.x) and (square.pos.y == except_pos.y )):
            square_criterion = False
        if(env.task=="give_n"):
            square_criterion = not square.picked_already
        if(env.task=="count_all_objects" or env.task=="touch_all_objects"):
            square_criterion = not square.touched_already

        if(square_criterion):
          d_curr = (env.hand.pos.x - square.pos.x)**2 + (env.hand.pos.y - square.pos.y)**2
          if((d_curr < d_min) and (square_criterion) ):
            d_min = d_curr
            square_n = n
            isFoundAny = True       
                
    return square_n, isFoundAny
  

def get_square_id_at_pos(env, pos):
  
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

        if( (square.pos.y == pos[1]) and (square.pos.x == pos[0]) ):
          square_n = n
          isFoundAny = True
                
    return square_n, isFoundAny
  

def move_to_square(env, n):
      
      squ_x = env.squares[n].pos.x
      squ_y = env.squares[n].pos.y
      
      move_to_pos = [squ_x, squ_y]      
      move_to(env, move_to_pos)
              
      
    
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


def pick_next_object(env):
    
    n, isFoundAny = find_next_object(env)
    move_to_square(env, n)
    env.update("pick")



def move_square_from_to(env, from_pos=[0,0], to_pos=[0,0]):
    
    #n, isFoundAny = get_square_id_at_pos(env, from_pos)
        
    #if(isFoundAny):        
        #move_to_square(env, n)
    move_to(env, from_pos)
    env.update("pick")
    move_to(env, to_pos)
    env.update("release")          

    #return (not isFoundAny)


def create_counting_sequence(env, n=-1):
  counting_sequence = []
  start_n=1
  
  if(n != -1):
    max_n = n
  else:
    max_n = env.n_squares
  
  for n in range(start_n,max_n+1):
      if(env.IsDecimal):
        n_string = int2base(n, base=env.base)
        if(env.leading_zero):
          n_string = "{:02d}".format(int(n_string))
        for s_i in n_string:
          counting_sequence.append(s_i)
      if(not env.IsDecimal):
          n_string = str(n)
          counting_sequence.append(n_string)
      #counting_sequence.append(10)
    #counting_sequence.append(11)

  return counting_sequence

def create_successor(env, n=-1):
  counting_sequence = []
  start_n=1
        #counting_sequence.append(10)
    #counting_sequence.append(11)

  return counting_sequence



def create_successor(env, n=-1):
  counting_sequence = []
  start_n=1
  
  if(n != -1):
    max_n = n
  else:
    max_n = env.n_squares
  
  if(env.IsDecimal):
    n_string = int2base(max_n+1, base=env.base)
    if(env.leading_zero):
      n_string = "{:02d}".format(int(n_string))
    for s_i in n_string:
      counting_sequence.append(s_i)
  if(not env.IsDecimal):
      n_string = str(max_n+1)
      counting_sequence.append(n_string)
      #counting_sequence.append(10)
    #counting_sequence.append(11)

  return counting_sequence


def int2base(x, base):
    digs = string.digits + string.ascii_letters
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)
    