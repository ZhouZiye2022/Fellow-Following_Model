import time
import numpy
import queue
import math
import networkx as nx
import random
import copy
DIM=2 #维度

def fast_norm(x):
  # Faster than np.linalg.norm
  return numpy.sqrt(x.dot(x))

class Agent:
  def __init__(self, Environment):
    self.pos = numpy.zeros(DIM)
    self.next_pos = numpy.zeros(DIM) 
    self.vel = numpy.zeros(DIM)
    self.next_vel  = numpy.zeros(DIM)
    self.ori_mean_angle = 0
    self.omega = 0
    self.desire_vel = numpy.zeros(DIM)
    self.ID = 0
    self.env = Environment
    self.timestep = self.env.timestep
    self.active = True
  
  def update_position(self, neighbors):
    #print("No update position defined for this class")
    pass
  
  def __str__(self):
    return f"{self.__class__.__name__}: pos: {self.pos} vel: {self.vel} active:{self.active}"
  __repr__ = __str__


class Fish(Agent):
    def __init__(self, Environment):
        #基本参数
        super().__init__(Environment)
        self.is_in_graph = 0
        self.graphID = 0

        self.target_point = self.env.bounds/2
        self.bodylength = 1
        self.near_neighbors = []
        self.sense_fishes = [] 
        self.sense_of_fishswarm = []
        self.neighbor_distance = 2
        self.repulsion = 2
        self.sense_of_predator = 50
        self.sense_of_other_fish = 4.0
        self.sense_of_fishswarm_length = 50
        
        self.sense_of_food = 50

        self.long_term_insecurity = 0    #长期不安全感
        self.short_term_insecurity = 0    #短期不安全感
        self.long_term_insecurity_orientation = numpy.zeros(DIM) #长期不安全感方向
        self.short_term_insecurity_orientation = numpy.zeros(DIM) #短期不安全感方向
        self.insecurity = 0    #不安全感
        self.cumulative_insecurity = 0    #累积不安全感
        self.threshold_insecurity = 10
        self.mode = 0
        self.insecurity_orientation = numpy.zeros(DIM) #不安全感方向
        self.max_insecurity = 1
        self.shannonEnt = 0.0                                #经验熵(香农熵)


        self.desvel = numpy.random.randn(DIM)
        self.last_desvel = numpy.zeros(DIM)
        self.dvel = numpy.zeros(DIM)
        self.last_vel = numpy.zeros(DIM)
        self.last2_vel = numpy.zeros(DIM)
        self.min_speed = 2
        self.max_speed = 3.0
        self.speed = self.min_speed
        self.roam_speed = self.max_speed * 0.75
        self.max_torque = 3.14
        self._180_PI = 180/math.pi
        self.blind_angle = 90.0                       #blind angle
        self.max_turning_rate = 60.0            #每秒最大转向速率


        self.random_time = 5*self.timestep
        self.turn_time = 50*self.timestep
        self.randvel = numpy.zeros(DIM)
        self.t = self.env.t 
        self.t1 = 0
        self.t2 = 0

    
    def random_swim(self):
        if (self.env.t - self.t1) > self.random_time:
            self.t1 = self.env.t
            self.randvel = self.desire_vel
            self.speed = self.max_speed * 0.75
            self.omega = self.max_turning_rate *0.75
            #print(self.t1 , self.env.t)
            #print(self.vel)
        else:
            self.desire_vel = self.randvel 
            #print(self.vel)
        if self.t1 == 0:
            self.desire_vel = numpy.random.randn(DIM)
            self.desire_vel = self.desire_vel * self.roam_speed/fast_norm(self.desire_vel) 
            self.speed = self.max_speed * 0.75
            self.omega = self.max_turning_rate *0.75
        if fast_norm(self.desire_vel) == 0:
            self.desire_vel = numpy.random.randn(DIM)
    def update_insecurity_and_position(self):
        #self.sensing_and_graph()
        #计算短期不安全感
        self.get_short_term_insecurity()
        #计算长期不安全感
        #self.long_term_insecurity = self.env.graphlist[self.graphID].long_term_insecurity  
        # Apply forces
        #self.insecurity = (self.long_term_insecurity + self.short_term_insecurity)/2  #不安全感
        self.insecurity = self.short_term_insecurity
        if self.insecurity > 0.5:
            self.cumulative_insecurity += self.insecurity * self.timestep
        #print(self.long_term_insecurity , self.short_term_insecurity)
        self.insecurity_orientation = - self.short_term_insecurity_orientation #不安全感方向
        #print('self.insecurity ',self.insecurity )
        #print('long_term_insecurity ',self.long_term_insecurity,'self.short_term_insecurity' ,self.short_term_insecurity)
        

        self.fishmoving()

    def environment_force(self):
        """
        Return the force exerted by the boundaries
        """
        force = numpy.zeros(DIM)

        #print(self.env.bounds - self.pos)
        if abs(self.pos[0] - self.env.bounds[0]/2) >  (self.env.bounds[0]/2 - 2):
            a = 1 / (1 + pow(-self.pos[0], 2))
            b = 1 / (1 + pow(self.env.bounds[0] - self.pos[0], 2))
            force[0] = 100 * (a - b)
        if abs(self.pos[1] - self.env.bounds[1]/2) >  (self.env.bounds[1]/2 - 2):
            a = 1 / (1 + pow(-self.pos[1], 2))
            b = 1 / (1 + pow(self.env.bounds[1] - self.pos[1], 2))
            force[1] = 100 * (a - b)

        return force
        




    def get_short_term_insecurity(self):
        self.near_neighbors = []
        self.sense_fishes = []  
        self.sense_of_fishswarm =[]      
                     
        for n in self.env.fish_group:
            #print(len(self.env.graphlist[self.graphID]),self.graphID)
            if n == 0:
                continue
            #print(n)
            distance = fast_norm(self.pos - n.pos)
            if  (distance < self.sense_of_other_fish and distance != 0):
                self.sense_fishes.append(n)
            

            if distance < self.sense_of_fishswarm_length and distance != 0:
                self.sense_of_fishswarm.append(n)
                


    def detect_boundary(self,lim):
        x= 0
        y= 0
        temp_desire_vel = numpy.zeros(DIM) 
        
        
        x_near_bound = 0
        y_near_bound = 0
        x = self.pos[0]
        y = self.pos[1]
        
        if x < lim or x > (self.env.bounds[0] -lim) :
            x_near_bound = 1
            self.desvel = numpy.zeros(DIM)  
        if y < lim or y > (self.env.bounds[1] -lim) :
            y_near_bound = 1    
            self.desvel = numpy.zeros(DIM) 
        if (x_near_bound + y_near_bound)  != 0:
            self.shannonEnt = 10
            if y_near_bound == 1:
                #temp_desire_vel[1] = -100 * numpy.sign( (self.env.bounds[1])/2 - x)
                temp_desire_vel[1] = 100/( (self.env.bounds[1])/2 - y)
                #self.speed = self.min_speed
                #self.omega = self.max_turning_rate /10
                self.desvel[1] = temp_desire_vel[1]  
            if x_near_bound == 1 :
                #temp_desire_vel[0] = -100 * numpy.sign( (self.env.bounds[0])/2 - y)
                temp_desire_vel[0] = 100/( (self.env.bounds[0])/2 - x)
                #self.speed = self.min_speed        
                #self.omega = self.max_turning_rate /10 
                self.desvel[0] = temp_desire_vel[0]     
            #print('self.pos,self.desvel,temp_desire_vel',self.pos,self.desvel,temp_desire_vel)
        #print(self.desvel )
        #print(self.env.bounds - self.pos)
    def detect_boundary2(self):
            x= 0
            y= 0
            x = self.pos[0]
            y = self.pos[1]
            
            if x < 0 :
                self.pos[0] = 0.5
            if x > self.env.bounds[0] :
                self.pos[0] = self.env.bounds[0]- 0.5
            if y < 0 :
                self.pos[1] = 0.5
            if y > self.env.bounds[1] :
                self.pos[1] = self.env.bounds[1] - 0.5
            
            #print(self.env.bounds - self.pos)

    def fishmoving(self):
        #print(self.env.predator_moving,fast_norm(self.pos - self.env.pred_pos),self.env.pred_pos)




        delta_x = numpy.zeros(DIM)
        repulsion_force, alliance_force, attraction_force,dvel = self.get_forces()
        #print(repulsion_force, alliance_force, attraction_force,dvel)
        self.detect_boundary(lim = 4)
        if self.speed == 0:
            self.speed = 0.00001     
        
        ori = self.vel/fast_norm(self.vel)
        if self.env.predator_moving == 1 and fast_norm(self.pos - self.env.pred_pos) < self.sense_of_predator:
            #print('--------------')
            pred_omg =0
            self.speed = self.max_speed *3
            #print(self.get_angle_from_a_to_b(self.vel,self.desvel))
            #pred_ori = numpy.zeros(DIM)
            pred_ori = self.env.predator.ori.copy()
            pred_dist = (self.pos - self.env.pred_pos)/fast_norm(self.pos - self.env.pred_pos)
            pred_omg = self.get_angle_from_a_to_b(pred_ori,pred_dist)
            if pred_omg <= 0:
                self.rotate_vector(pred_ori, -90)
            else:
                self.rotate_vector(pred_ori, 90)
            self.desvel =  pred_dist+pred_ori
            #print(self.get_angle_from_a_to_b(self.vel,self.desvel))
        #self.omega =  -2* self.speed/20
        omg = self.get_angle_from_a_to_b(self.vel,self.desvel)
        #print(self.vel,self.desvel,omg)
        #print(self.get_angle_from_a_to_b(self.vel,self.desvel))
        omg = self.get_angle_from_a_to_b(self.vel,(self.desvel+0.5*self.last_desvel))
        self.last_desvel =self.desvel                 
        #print('self.vel,self.desvel',self.vel,self.desvel)
        if abs(omg) > self.max_turning_rate:
            omg = numpy.sign(omg)*self.max_turning_rate * self.timestep
        #print('-------------------------------')
        delta_x = ori * self.speed * self.timestep
        #print('self.pos, ori,,self.desvel,omg',self.pos, ori,self.desvel,omg)
        self.rotate_vector(delta_x, omg)
        self.next_pos = self.pos + delta_x
        #self.pos = self.pos + delta_x
        self.detect_boundary2()

        self.next_vel = delta_x/fast_norm(delta_x)
        #self.vel = delta_x/fast_norm(delta_x)
            #print('omg,delta_x',omg,delta_x)
        #print(self.vel)       
    
        #self.dvel = self.vel - self.last_vel
        #self.last2_vel = self.last_vel
        #self.last_vel = self.vel
        ori_angles =[]
        for n in self.sense_fishes:
            dist1 =  n.pos - self.pos
            distance1 = fast_norm(dist1)
            if distance1 < 4:

                ori_angle = self.get_angle_from_a_to_b(self.vel, n.vel)
                ori_angles.append(ori_angle)
        if len(ori_angles) ==0:
            self.ori_mean_angle = 0
        else:
            #print(len(ori_angles)  )
            self.ori_mean_angle = sum(ori_angles)/len(ori_angles)        
        

         


    def rotate_vector(self, vector, degrees):
        x = vector[0]
        y = vector[1] 
        
        temp_x = ( (x * math.cos(degrees / self._180_PI)) - (y * math.sin(degrees / self._180_PI)) )
        temp_y = ( (x * math.sin(degrees / self._180_PI)) + (y * math.cos(degrees / self._180_PI)) )         
        vector[0] = temp_x
        vector[1] = temp_y
        


    def get_forces(self):
        repulsion_force = numpy.zeros(DIM)
        alliance_force = numpy.zeros(DIM)
        attraction_force = numpy.zeros(DIM)
        dvel = numpy.zeros(DIM)
        _is_repulsion = 0
        _num_alliance = 0
        _last_distance = 0
        _repulsion_in = 0
        _forward_length = 0
        _forward_speed_count = 0
        dist_sum = numpy.zeros(DIM) 

        forward_vel_sum = numpy.zeros(DIM) 
        left_vel_sum = numpy.zeros(DIM) 
        right_vel_sum = numpy.zeros(DIM) 
        forward_desvel = numpy.zeros(DIM) 
        left_desvel = numpy.zeros(DIM) 
        right_desvel = numpy.zeros(DIM) 
        k1 = 0
        k2 = 0
        k3 = 0 

        forward_pos_sum = numpy.zeros(DIM) 
        forward_pos = numpy.zeros(DIM)
        forward_count = 0
        forward_speed_sum = 0
        forward_omega_sum = 0
        forward_speed = 0
        forward_omega= 0                
        left_count = 0
        left_speed_sum = 0
        left_omega_sum = 0
        left_speed = 0
        left_omega = 0  
        left_dist_sum =  0
        left_dist =  0                
        right_count = 0
        right_speed_sum = 0
        right_omega_sum = 0 
        right_dist_sum =  0
        right_dist =  0   
        right_speed = 0
        right_omega = 0  
        _forward_obs = 0
        _left_obs = 0
        _right_obs = 0
        num_dist_sum_left = 0
        num_dist_sum_right = 0
        dist_sum_left = numpy.zeros(DIM) 
        dist_sum_right = numpy.zeros(DIM) 

        right_angle = 0
        right_projection = numpy.zeros(DIM) 
        right_vector =  numpy.zeros(DIM) 
        left_angle =0
        left_projection = numpy.zeros(DIM) 
        left_vector = numpy.zeros(DIM) 
        norm_vel = fast_norm(self.vel)

        attraction = numpy.zeros(DIM) 
        attraction_num = 0
        attraction_mean =  numpy.zeros(DIM) 
        repulsion = numpy.zeros(DIM) 
        repulsion_num = 0
        repulsion_mean   =  numpy.zeros(DIM)       
        forward_dist_sum = numpy.zeros(DIM) 
        forward_dist =  numpy.zeros(DIM) 
        target_dist = numpy.zeros(DIM) 
        target_distance = 0
        target_angle = 0
        target_angle_delta = 0
        target_omega = 0  
        max_ori_angle = 0
        min_ori_angle = 0
        temp_ori_angle  = 0             
        angles = []
        ori_angles = []
        angle_probable = [0]*12

        #计算香农熵
        for n in self.sense_fishes:
            dist1 =  n.pos - self.pos
            distance1 = fast_norm(dist1)
            if distance1 < 4:

                ori_angle = self.get_angle_from_a_to_b(self.vel, n.vel)
                ori_angles.append(ori_angle)
        if len(ori_angles) ==0:
            self.ori_mean_angle = 0
        else:
            self.ori_mean_angle = sum(ori_angles)/len(ori_angles)

        len_angles = len(ori_angles)    
        for angle in ori_angles:
            if  0 <= angle < 30:
                angle_probable[0] += 1/len_angles
            if 30 <= angle < 60:
                angle_probable[1] += 1/len_angles               
            if 60 <= angle < 90:
                angle_probable[2] += 1/len_angles
            if 90 <= angle < 120:
                angle_probable[3] += 1/len_angles
            if 120 <= angle < 150:
                angle_probable[4] += 1/len_angles     
            if 150 <= angle <= 180:
                angle_probable[5] += 1/len_angles                              
            if -30 <= angle < 0:
                angle_probable[6] += 1/len_angles
            if -60 <= angle < -30:
                angle_probable[7] += 1/len_angles               
            if -90 <= angle < -60:
                angle_probable[8] += 1/len_angles
            if -120 <= angle < -90:
                angle_probable[9] += 1/len_angles
            if -150 <= angle < -120:
                angle_probable[10] += 1/len_angles     
            if -180 <= angle < -150:
                angle_probable[11] += 1/len_angles  

        self.shannonEnt = 0.0                                #经验熵(香农熵)
        for prob in angle_probable:                         #计算香农熵
            if prob != 0:
                self.shannonEnt -= prob * math.log(prob, 2)           #利用公式计算
        xangle = 0
        xangles =[]
  
        if  len(self.sense_fishes) <4:
            self.shannonEnt = 10  
        if  self.shannonEnt > -1.2:
            for n in self.sense_fishes:
                orient = n.vel
                
                dist =  n.pos - self.pos
                distance = fast_norm(dist)
                ori_angle = self.get_angle_from_a_to_b(self.vel, n.vel)
                angle = self.get_angle_from_a_to_b(self.vel, dist)
                xangle = numpy.cross(self.vel, dist)
                xangles.append(xangle)       
                if  self.repulsion+ 5*(self.sense_of_other_fish - self.repulsion)/6 < distance < self.sense_of_other_fish:
                    if 30< abs(angle) < 150:
                        attraction += dist
                        attraction_num += 1
                if  self.repulsion/2 > distance:
                        repulsion += -dist
                        repulsion_num += 1   
                          
                #print(n.pos , self.pos,self.vel, dist,angle)
                
                #self.rotate_vector(self.vel, angle)
                #print(n.pos , self.pos,self.vel, dist,angle)
                #print('-----------------')
                #angle = math.atan2(orient[1], orient[0]) * self._180_PI
                ori_angle = 0
                if  0 <= angle < 30 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        #self.speed = 0
                        self.omega = 0
                        _forward_obs = 1
                    else:
                        forward_count += 1
                        forward_speed_sum += n.speed
                        forward_omega_sum += n.omega
                        forward_vel_sum += n.vel
                        forward_pos_sum += n.pos
                        forward_dist_sum += dist                        
                if 30 <= angle < 60 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1 
                    else:
                        left_count += 1
                        left_speed_sum += n.speed
                        left_omega_sum += n.omega 
                        left_vel_sum += n.vel
                        left_dist_sum += abs(math.sin(angle))                        
                if 60 <= angle < 90 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1
                    else:
                        left_count += 1
                        left_speed_sum += n.speed
                        left_omega_sum += n.omega 
                        left_vel_sum += n.vel
                        left_dist_sum += abs(math.sin(angle))                                                               
                if 90 <= angle < 120 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1 
                    else:
                        left_count += 1
                        left_speed_sum += n.speed
                        left_omega_sum += n.omega 
                        left_vel_sum += n.vel
                        left_dist_sum += abs(math.sin(angle))                                                            
                if 120 <= angle < 150 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1 
                    else:
                        left_count += 1
                        left_speed_sum += n.speed
                        left_omega_sum += n.omega 
                        left_vel_sum += n.vel
                        left_dist_sum += abs(math.sin(angle))                                                                  
                if 150 <= angle <= 180:
                    pass                              
                if -30 <= angle < 0 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        #self.speed = 0
                        self.omega = 0 
                        _forward_obs = 1    
                    else:
                        forward_count += 1
                        forward_speed_sum += n.speed
                        forward_omega_sum += n.omega
                        forward_vel_sum += n.vel
                        forward_pos_sum += n.pos 
                        forward_dist_sum += dist                                                       
                if -60 <= angle < -30 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                      
                    else:
                        right_count += 1
                        right_speed_sum += n.speed
                        right_omega_sum += n.omega
                        right_vel_sum += n.vel
                        right_dist_sum += abs(math.sin(angle))                                                     
                if -90 <= angle < -60 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1  
                    else:
                        right_count += 1
                        right_speed_sum += n.speed
                        right_omega_sum += n.omega
                        right_vel_sum += n.vel
                        right_dist_sum += abs(math.sin(angle))                                                           
                if -120 <= angle < -90 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                
                    else:
                        right_count += 1
                        right_speed_sum += n.speed
                        right_omega_sum += n.omega
                        right_vel_sum += n.vel
                        right_dist_sum += abs(math.sin(angle))                                             
                if -150 <= angle < -120 and abs(ori_angle) < 90:

                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1    
                    else:
                        right_count += 1
                        right_speed_sum += n.speed
                        right_omega_sum += n.omega
                        right_vel_sum += n.vel
                        right_dist_sum += abs(math.sin(angle))                                                             
                if -180 <= angle < -150:
                    pass            
            #print(forward_count,left_count,right_count)
            #print(sum(xangles)/len(xangles),sum(xangles),len(xangles))
            if len(self.sense_fishes) == 0:
                if len(self.sense_of_fishswarm) == 0:
                    self.random_swim()
                    self.desvel =self.desire_vel
                else:
                    
                    for n in self.sense_of_fishswarm:
                        dist =  n.pos - self.pos
                        angle = self.get_angle_from_a_to_b(self.vel, dist)
                        if 0 <= angle <100:
                            dist_sum_left += dist
                            num_dist_sum_left += 1
                        if 0 > angle > -100:
                            dist_sum_right += dist 
                            num_dist_sum_right += 1 
                        if num_dist_sum_left == 0:
                            num_dist_sum_left = 1
                        if num_dist_sum_right == 0:
                            num_dist_sum_right = 1
                        if fast_norm(dist_sum_right/num_dist_sum_right) < fast_norm(dist_sum_left/num_dist_sum_left) :
                            dist_sum = dist_sum_left - dist_sum_right
                        else:
                            dist_sum = dist_sum_right - dist_sum_left

                    if fast_norm(dist_sum) == 0:
                        self.random_swim()
                        dist_sum = self.desvel   
                    self.speed = self.max_speed *2 
                    self.desvel = dist_sum

            else:
                if forward_count != 0:
                    forward_speed = forward_speed_sum/forward_count
                    forward_omega = forward_omega_sum/forward_count
                    forward_desvel = forward_vel_sum/forward_count
                    forward_pos = forward_pos_sum/forward_count
                    forward_dist = forward_dist_sum/forward_count
                    target_ori = forward_pos - self.pos
                    dist = fast_norm(target_ori)
                    dot_product = numpy.dot(self.vel, target_ori)/(norm_vel * dist)
                    if dot_product > 1:
                        dot_product = 1
                    if dot_product <-1:
                        dot_product = -1
                    angle_target_ori = math.acos(dot_product) * self._180_PI
                    sign = numpy.sign(numpy.cross(self.vel, target_ori))
                    if sign == 0:
                        sign = 0.000000001*numpy.random.randn(1)
                    angle_target_ori = sign * angle_target_ori
                    
                    dt= dist/forward_speed
                    omega = angle_target_ori/dt
                    forward_omega =omega
                    #print('dist,forward_speed,angle_target_ori',dist,forward_speed,angle_target_ori)
                    #print('forward_omega',forward_omega)
                    #forward_omega = -1* forward_speed/20
                    
                if left_count != 0:
                    left_speed = left_speed_sum/left_count
                    left_omega = left_omega_sum/left_count
                    left_desvel = left_vel_sum/left_count
                    left_dist = left_dist_sum/left_count
                    left_angle = self.get_angle_from_a_to_b(self.vel,left_desvel)
                    if norm_vel != 0:
                        left_projection = self.vel * math.cos(left_angle)*fast_norm(left_desvel)/norm_vel
                    left_vector = left_desvel - left_projection
                if right_count != 0:
                    right_speed = right_speed_sum/right_count
                    right_omega = right_omega_sum/right_count 
                    right_desvel = right_vel_sum/right_count
                    right_dist = right_dist_sum/right_count
                    right_angle = self.get_angle_from_a_to_b(self.vel,right_desvel)
                    if norm_vel != 0:
                        right_projection = self.vel * math.cos(right_angle)*fast_norm(right_desvel)/norm_vel
                    right_vector = right_desvel - right_projection
                #print(right_vector,left_vector,norm_vel)
                if attraction_num != 0:
                    attraction_mean = attraction/attraction_num      
                if repulsion_num != 0:              
                    repulsion_mean =  repulsion/repulsion_num  
               
                if (forward_count + left_count + right_count) == 0:
                    if self.env.is_traget == 1:
                        self.desvel =  self.env.traget - self.pos
                    else:
                        self.random_swim()
                        self.desvel = self.desire_vel
                    #print('self.desvel0',self.desvel)
                    if len(self.sense_of_fishswarm) != 0:
                        #print(_forward_obs + _left_obs + _right_obs, len(self.sense_fishes),len(self.sense_of_fishswarm))
                        for n in self.sense_of_fishswarm:
                            dist =  n.pos - self.pos
                            angle = self.get_angle_from_a_to_b(self.vel, dist)
                            if 0 <abs(angle) < 120:
                                dist_sum += dist
                        if fast_norm(dist_sum) != 0:
                            self.desvel =  dist_sum                      
                   

                else:
                    if forward_count != 0:
                        k0 = 0
                        k1 = 1
                        k2 = 0
                        k3 = 0                       
                        if (left_count + right_count) != 0:
                            k0 = 0
                            k1 = 1
                            k2 = 1
                            k3 = 1
                        else:
                            if left_count == 0 and right_count != 0:
                                k0 = 0
                                k1 = 1
                                k2 = 1
                                k3 = 0
                            if left_count != 0 and right_count == 0:
                                k0 = 0
                                k1 = 1
                                k2 = 0
                                k3 = 1 
                    elif (left_count + right_count) != 0:
                        k0 = 0.1
                        k1 = 0
                        k2 = 1
                        k3 = 1
                        #self.omega =  -1* forward_speed/20
                        if  left_count == 0 and right_count != 0:
                            k0 = 0.1
                            k1 = 0
                            k2 = 1
                            k3 = 0
                        if left_count != 0 and right_count == 0:
                            k0 = 0.1
                            k1 = 0
                            k2 = 0
                            k3 = 1 
                        #print(left_count ,right_count)
                        if left_count <= 3 and right_count <= 3 :
                        #print(len(self.sense_of_fishswarm))
                            for n in self.sense_of_fishswarm:
                                dist =  n.pos - self.pos
                                angle = self.get_angle_from_a_to_b(self.vel, dist)
                                if 0 <= angle < 60:
                                    dist_sum_left += dist
                                    num_dist_sum_left += 1
                                if 0 > angle > -60:
                                    dist_sum_right += dist 
                                    num_dist_sum_right += 1 
                                if num_dist_sum_left == 0:
                                    num_dist_sum_left = 1
                                if num_dist_sum_right == 0:
                                    num_dist_sum_right = 1
                                #print(fast_norm(dist_sum_right/num_dist_sum_right) , fast_norm(dist_sum_left/num_dist_sum_left) )
                                if fast_norm(dist_sum_right/num_dist_sum_right) < fast_norm(dist_sum_left/num_dist_sum_left) :
                                    dist_sum = dist_sum_left - dist_sum_right
                                else:
                                    dist_sum = dist_sum_right - dist_sum_left

                            if fast_norm(dist_sum) == 0:
                                self.random_swim()
                                dist_sum = self.desvel
                                if self.env.is_traget == 1:
                                    dist_sum =  self.env.traget - self.pos
                            #print(dist_sum,self.desvel)
                            k0 = 1
                            k1 = 0
                            k2 = 1
                            k3 = 1
                            #k0 = 0

                                                    
                    #print(k1,k2,k3)
                    target_dist = self.target_point - self.pos
                    target_distance = fast_norm(target_dist)
                    target_angle = self.get_angle_from_a_to_b(self.vel, target_dist)
                    target_angle_delta = abs(target_angle) - 90
                    target_omega = self.max_speed/target_distance
                                       
                    if target_angle_delta > 0:
                        target_omega = - (1+target_angle_delta/90)*target_omega
                    if target_angle_delta < 0:
                        target_omega =  (1+target_angle_delta/90)*target_omega
                    k4 = 1                    
                    self.speed = (self.max_speed *2 * (k0*10) +forward_speed * k1 + right_speed *k2 + left_speed *k3) /(k0+k1+k2+k3)
                    #self.speed = self.max_speed *2 * (k0*10) +forward_speed * k1

                    self.omega = (forward_omega * k1 + right_omega *k2 + left_omega *k3+target_omega ) /(k1+k2+k3+k4)
                    self.desvel = forward_dist + (k0*dist_sum+forward_desvel * k1 + right_desvel *k2 + left_desvel *k3 +attraction_mean/20 +repulsion_mean/10) /(k1+k2+k3 + k0 + 1/30 +1/10)
                    self.desvel = forward_dist + (k0*dist_sum+ right_desvel *1*k2 + left_desvel *1*k3 +attraction_mean/30 +repulsion_mean*0.1) /(k2+k3 + k0 + 1/30 +1/5)
                    #self.desvel = forward_dist + (right_vector *0.5*k2 + left_vector *0.5*k3 +attraction_mean/20 +repulsion_mean/5) /(0.5*k2+0.5*k3 + 1/30 +1/10)


                    
                    #print(self.desvel)
                    #print('-------------')
                    #print(right_omega,left_omega)
                    '''
                    if forward_count == 0 and left_count == 0 and right_count ==1:
                        self.rotate_vector(self.desvel, 200 ) 
                    if forward_count == 0 and left_count == 1 and right_count ==0:
                        self.rotate_vector(self.desvel, -200 ) 
                        '''
                    #print('self.desvel1',self.desvel)
                    #right_dist
                    #self.desvel = 
                    '''
                if _forward_obs == 1:
                    self.speed = self.min_speed
                    self.omega = 0
                elif _left_obs == 1 and  _right_obs == 0 :
                    self.speed = self.max_speed
                    self.omega = -self.max_turning_rate /10
                    self.rotate_vector(self.desvel, self.omega )               
                elif _left_obs == 0 and  _right_obs == 1 :
                    self.speed = self.max_speed
                    self.omega = self.max_turning_rate /10
                    self.rotate_vector(self.desvel, self.omega ) 
                elif _left_obs == 1 and  _right_obs == 1 :
                    self.speed = self.max_speed
                    self.omega = 0   '''
                #print('self.desvel2',self.desvel)   

                if  (_forward_obs + _left_obs + _right_obs) > 0:
                    self.speed = (self.min_speed * _forward_obs + self.max_speed * (_left_obs + _right_obs))/(_forward_obs + _left_obs + _right_obs)
                    self.omega = -self.max_turning_rate * _left_obs + self.max_turning_rate * _right_obs
                    
                    if _forward_obs == 0 :
                        max_ori_angle = max(ori_angles)
                        min_ori_angle = min(ori_angles)
                        self.speed = 1* self.speed 
                        if abs(max_ori_angle) > abs(min_ori_angle):
                            temp_ori_angle = max_ori_angle
                            self.omega = 1*temp_ori_angle -self.max_turning_rate * _left_obs + self.max_turning_rate * _right_obs
                        else:
                            temp_ori_angle = min_ori_angle  
                            self.omega = 1*temp_ori_angle -self.max_turning_rate * _left_obs + self.max_turning_rate * _right_obs
                                         
                    self.rotate_vector(self.desvel, self.omega/3) 
                    
            if self.speed < self.min_speed:
                self.speed = self.min_speed
            if self.speed > 3*self.max_speed:
                self.speed = 3*self.max_speed                   
                #print(_forward_obs , _left_obs , _right_obs,self.speed,self.desvel)
                #print('--------------')
            
            
        else:
            for n in self.sense_fishes:
                orient = n.vel
                dist =  n.pos - self.pos
                distance = fast_norm(dist)

                angle = self.get_angle_from_a_to_b(self.vel, dist)
                #print(n.pos , self.pos,self.vel, dist,angle)
            
                #self.rotate_vector(self.vel, angle)
                #print(n.pos , self.pos,self.vel, dist,angle)
                #print('-----------------')
                #angle = math.atan2(orient[1], orient[0]) * self._180_PI
                if  0 <= angle < 30:
                    forward_count += 1
                    forward_speed_sum += n.speed
                    forward_omega_sum += n.omega
                    forward_pos_sum += n.pos
                    forward_vel_sum += n.vel
                    if distance < self.repulsion:
                        #self.speed = 0
                        self.omega = 0
                        _forward_obs = 1
                if 30 <= angle < 60:
                    left_count += 1
                    left_speed_sum += n.speed
                    left_omega_sum += n.omega 
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1 
                if 60 <= angle < 90:
                    left_count += 1
                    left_speed_sum += n.speed
                    left_omega_sum += n.omega
                    left_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1                                    
                if 90 <= angle < 120:
                    left_count += 1
                    left_speed_sum += n.speed
                    left_omega_sum += n.omega 
                    left_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1                                  
                if 120 <= angle < 150:
                    left_count += 1
                    left_speed_sum += n.speed
                    left_omega_sum += n.omega 
                    left_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = -self.max_turning_rate
                        _left_obs = 1                                        
                if 150 <= angle <= 180:
                    pass                              
                if -30 <= angle < 0:
                    forward_count += 1
                    forward_speed_sum += n.speed
                    forward_omega_sum += n.omega
                    forward_pos_sum += n.pos 
                    forward_vel_sum += n.vel
                    if distance < self.repulsion:
                        #self.speed = 0
                        self.omega = 0 
                        _forward_obs = 1                               
                if -60 <= angle < -30:
                    right_count += 1
                    right_speed_sum += n.speed
                    right_omega_sum += n.omega
                    right_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                                                   
                if -90 <= angle < -60:
                    right_count += 1
                    right_speed_sum += n.speed
                    right_omega_sum += n.omega 
                    right_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                                   
                if -120 <= angle < -90:
                    right_count += 1
                    right_speed_sum += n.speed
                    right_omega_sum += n.omega 
                    right_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                                   
                if -150 <= angle < -120:
                    right_count += 1
                    right_speed_sum += n.speed
                    right_omega_sum += n.omega
                    right_vel_sum += n.vel
                    if distance < self.repulsion:
                        self.speed = self.max_speed
                        self.omega = self.max_turning_rate
                        _right_obs = 1                                         
                if -180 <= angle < -150:
                    pass            
                angles.append(angle)
                distance = fast_norm(self.pos - n.pos)
            #print(forward_count,left_count,right_count)
            
            if len(self.sense_fishes) == 0:
                self.random_swim()
            else:
                if forward_count != 0:
                    forward_speed = forward_speed_sum/forward_count
                    forward_omega = forward_omega_sum/forward_count
                    forward_pos = forward_pos_sum/forward_count
                    forward_desvel = forward_vel_sum/forward_count
                    target_ori = forward_pos - self.pos
                    dist = fast_norm(target_ori)
                    dot_product = numpy.dot(self.vel, target_ori)/(fast_norm(self.vel) * dist)
                    if dot_product > 1:
                        dot_product = 1
                    if dot_product <-1:
                        dot_product = -1
                    angle_target_ori = math.acos(dot_product) * self._180_PI
                    sign = numpy.sign(numpy.cross(self.vel, target_ori))
                    if sign == 0:
                        sign = 0.000000001*numpy.random.randn(1)
                    angle_target_ori = sign * angle_target_ori
                    
                    dt= dist/forward_speed
                    omega = angle_target_ori/dt
                    forward_omega =omega
                    #print('dist,forward_speed,angle_target_ori',dist,forward_speed,angle_target_ori)
                    #print('forward_omega',forward_omega)
                    #forward_omega = -1* forward_speed/20
                    if left_count != 0:
                        left_speed = left_speed_sum/left_count
                        left_omega = left_omega_sum/left_count
                        left_desvel = left_vel_sum/left_count
                    if right_count != 0:
                        right_speed = right_speed_sum/right_count
                        right_omega = right_omega_sum/right_count 
                        right_desvel = right_vel_sum/right_count
                
                
                if (forward_count + left_count + right_count) == 0:
                    self.random_swim()
                    self.desvel = self.desire_vel 
                else:
                    if forward_count != 0:
                        k0 = 0
                        k1 = 1
                        k2 = 0
                        k3 = 0                       
                        if (left_count + right_count) != 0:
                            k0 = 0
                            k1 = 1
                            k2 = 0.1
                            k3 = 0.1
                        else:
                            if left_count == 0 and right_count != 0:
                                k0 = 0
                                k1 = 1
                                k2 = 0.2
                                k3 = 0
                            if left_count != 0 and right_count == 0:
                                k0 = 0
                                k1 = 1
                                k2 = 0
                                k3 = 0.2 
                    elif (left_count + right_count) != 0:
                        k0 = 1
                        k1 = 0
                        k2 = 1
                        k3 = 1
                        #self.omega =  -1* forward_speed/20
                        if  left_count == 0 and right_count != 0:
                            k0 = 1
                            k1 = 0
                            k2 = 1
                            k3 = 0
                        if left_count != 0 and right_count == 0:
                            k0 = 1
                            k1 = 0
                            k2 = 0
                            k3 = 1 
                    #print(k1,k2,k3)
                    target_dist = self.target_point - self.pos
                    target_distance = fast_norm(target_dist)
                    target_angle = self.get_angle_from_a_to_b(self.vel, target_dist)
                    target_angle_delta = abs(target_angle) - 90
                    target_omega = self.max_speed/target_distance
                                       
                    if target_angle_delta > 0:
                        target_omega = - (1+target_angle_delta/90)*target_omega
                    if target_angle_delta < 0:
                        target_omega =  (1+target_angle_delta/90)*target_omega
                    k4 = 1
                    self.speed = (self.max_speed * k0 +forward_speed * k1 + right_speed *k2 + left_count *k3 ) /(k0+k1+k2+k3)
                    self.omega = (forward_omega * k1 + right_omega *k2 + left_omega *k3 + target_omega) /(k1+k2+k3+k4)
                    self.desvel = (forward_desvel * k1 + right_desvel *k2 + left_desvel *k3 ) /(k1+k2+k3)
                    
                if _forward_obs == 1:
                    self.speed = self.min_speed
                    self.omega = 0
                elif _left_obs == 1 and  _right_obs == 0 :
                    self.speed = self.max_speed
                    self.omega = -self.max_turning_rate                
                elif _left_obs == 0 and  _right_obs == 1 :
                    self.speed = self.max_speed
                    self.omega = self.max_turning_rate 
                elif _left_obs == 1 and  _right_obs == 1 :
                    self.speed = self.max_speed
                    self.omega = 0 

        
        #print(forward_count,left_count,right_count)
        #print(self.speed, self.omega)

        #print(len(self.sense_of_fishswarm))
        
        #print('repulsion_force,alliance_force',fast_norm(repulsion_force),fast_norm(alliance_force))
        
        return repulsion_force, alliance_force , attraction_force,dvel

    def get_angle_from_a_to_b(self,a,b ):
        dot_product = numpy.dot(a, b)/(fast_norm(a) * fast_norm(b) )
        if dot_product > 1:
            dot_product = 1
        if dot_product <-1:
            dot_product = -1
        angle = math.acos(dot_product) * self._180_PI
        sign = numpy.sign(numpy.cross(a, b))
        if sign == 0:
            sign = 0.000000001*numpy.random.randn(1)
        angle = sign * angle  
        return angle   
'''

class Group:
    def __init__(self, num_fish=50, DIM=2, timestep = 0.5):
        self.timestep = timestep
        self.fish_group = []
        self.number = num_fish
        self.DIM = DIM
        self.mean_insecurity = 0
        self.long_term_insecurity = 0
        self.bounds = numpy.array([50, 50, 50]) ###############################修改
        if self.DIM == 2:
            self.bounds = self.bounds[:2]
        self.history_number = 10  #长期不安全感的记录步长
        self.history_mean_insecurity = queue.Queue(self.history_number)
        for _ in range(self.history_number):
            self.history_mean_insecurity.put(0)
        self.init_fishes()

    def init_fishes(self):
        for _ in range(self.number):
            agent = Fish(self)
          
            # Position agent uniformly in environment
            agent.pos = numpy.random.rand(self.DIM) * self.bounds
            
            self.fish_group.append(agent)

    def get_mean_insecurity(self):
        sum_insecurity = 0
        for i in self.number:            
            fishi = self.fish_group[i]
            sum_insecurity += fishi.get_short_term_insecurity()
        self.mean_insecurity = sum_insecurity/self.number

    def get_long_term_insecurity(self):
        self.history_mean_insecurity.get()
        self.history_mean_insecurity.put(self.mean_insecurity)
        self.long_term_insecurity = sum(self.history_mean_insecurity.queue)/self.history_number
        return self.long_term_insecurity
    
    def Group_move_timestep(self):
       
        for fish in  self.fish_group:
            fish.update_insecurity_and_position()
            

    def get_positions(self):
        positions = []
        all = self.fish_group
        for a in all:
            positions.append(a.pos)
        #print( all[2].pos)
        return numpy.array(positions)
    
    def get_velocity(self):
        vel = []
        all = self.fish_group 
        for a in all:
            vel.append(a.vel / fast_norm(a.vel))                   
        return numpy.array(vel)
'''
class Predator:
    def __init__(self, start_pos,end_pos,speed,DIM=2):
        self.speed = speed
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.ori = (end_pos-start_pos)/fast_norm(end_pos-start_pos)
        self.pos = numpy.zeros(DIM)  

    def run(self,dt):
        self.pos = self.start_pos + self.ori *dt*self.speed
        #print(self.pos,dt,fast(self.ori))
        return self.pos


class Environment:

    def __init__(self,x=50,y=50,z=50, DIM=2, num_fish=50, timestep = 0.5):
        
        #设置环境
        self.DIM = DIM
        self.bounds = numpy.array([x, y, z]) ###############################修改
        if self.DIM == 2:
            self.bounds = self.bounds[:2]
        self.num_fish = num_fish
        self.DIM = DIM
        #时间参数
        self.timestep = timestep
        self.t = 0
        #初始化Fishes
        self.fish_group = []
        self.graphlist = []
        G = nx.Graph()
        self.graphlist.append(G) 

        #Predator
        self.pred_pos = numpy.zeros(DIM)
        self.pred_start_pos = numpy.zeros(DIM)
        self.pred_end_pos = numpy.zeros(DIM)
        self.pred_start_pos[0] = 100
        self.pred_start_pos[1] = 100
        self.pred_end_pos[0] = 50
        self.pred_end_pos[1] = 1
        self.predator =  Predator( start_pos = self.pred_start_pos,end_pos =self.pred_end_pos ,speed = 20,DIM=2)
        self.predator_moving = 0      

        self.is_traget = 0
        self.traget = numpy.zeros(DIM)         
        #更新时间步
        #self.Group_of_fishes = Group(self.num_fish , self.DIM , self.timestep)
        self.history_number = 10  #长期不安全感的记录步长
        nx.Graph.history_mean_insecurity = queue.Queue(self.history_number)
        nx.Graph.mean_insecurity = 0  
        nx.Graph.long_term_insecurity = 0
        self.init_fishes()



    def init_fishes(self):
        for i in range(self.num_fish):
            agent = Fish(self)
            or_x=0
            or_y = 0
            e_angle = 30
            r0 = 12
            a0= 11
            b0= 11
            d = 2
            n = 30
            quan = 3
            mod = i%(quan)
            if mod == 0:
                kd = 0
            if mod == 1:
                kd = 1
            if mod == 2:
                kd = 2                
            r = r0 + kd * d
            a = a0 + kd * d
            b = b0 + kd * d
            theta = 2*math.pi/n * int(i/quan)
            #yuan
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            xy = [dx,dy]
            orixy = numpy.random.rand(self.DIM)
            orixy[0] = dy
            orixy[1] = -dx
            #tuoyuan
            or_x = a * math.cos(theta)
            or_y = b * math.sin(theta)
            length_or = math.sqrt(or_x * or_x + or_y * or_y)
            or_theta = math.atan2(or_y, or_x)
            new_theta = or_theta + e_angle/180*numpy.pi
            new_x = length_or * math.cos(new_theta)
            new_y = length_or * math.sin(new_theta)
            xy = [new_x,new_y]
            orixy = numpy.random.rand(self.DIM)
            orixy[0] = -new_y
            orixy[1] = new_x

            agent.pos = self.bounds/2 + xy+ 0.1*2*numpy.random.randn(2)
            agent.vel = 3*orixy/numpy.linalg.norm(orixy)
            #agent.omega =  -2* 2.25/r
            agent.ID = i
            #agent.pos = self.bounds/4+numpy.random.rand(self.DIM) * self.bounds/4
            #agent.vel = numpy.random.rand(self.DIM)/10  
            

            #######直线
            quan2 = 1
            mod = i%(quan2)
            if mod == 0:
                krect = 0.5
            if mod == 1:
                krect = 1.5
            if mod == 2:
                krect = -1.5   
            if mod == 3:
                krect = -0.5            
            init_pos = numpy.zeros(2)
            init_pos[0] = self.bounds[0]/10 + int(i/quan2)*2
            init_pos[1] = self.bounds[1]/2 + 0*krect*2

            #print(init_pos.shape)
            x_num =30
            coordinates =[]
            #for i in range(self.num_fish):
            a = int(i/x_num)
            b = i - a*x_num
            init_pos = [self.bounds[0]/10 +2.5*b , self.bounds[1]/2+2.5*a ] + 0.0001*2*numpy.random.randn(2)
            #print(init_pos)
            #print(init_pos.shape)
            #coordinates.append( [av_dist*b , av_dist*a ] )
            #coordinates.append( [2*b , 2*a ] + 0.1*2*numpy.random.randn(1,2))
            #print(coordinates)
            #coordinates = numpy.array(coordinates).reshape(len(coordinates),2)                
            init_vel = numpy.zeros(2) 
            init_vel[0] =  1   
            init_fenbu = numpy.zeros(2)      
            init_fenbu[0] = self.bounds[0]
            init_fenbu[1] = self.bounds[1]/3 
            
            #agent.vel = numpy.random.randn(self.DIM)/10  
            #agent.pos = numpy.random.rand(self.DIM) * self.bounds
            #agent.vel = numpy.random.randn(self.DIM)  
            #agent.pos = init_pos+numpy.random.randn(self.DIM) * self.bounds/20
            #agent.vel = init_vel + numpy.random.randn(self.DIM)/10   
            #print(init_pos,init_vel)
            agent.pos = init_pos
            agent.vel = init_vel     
            agent.pos = numpy.random.rand(self.DIM) * self.bounds
            agent.vel = numpy.random.randn(self.DIM)    
                              
            self.fish_group.append(agent)            
            
            '''
            # Position agent uniformly in environment
            agent.pos = self.bounds/4+numpy.random.rand(self.DIM) * self.bounds/4
            agent.vel = numpy.random.rand(self.DIM)/10
            agent.ID = i
            self.fish_group.append(agent)
            '''
            
    def get_ellipse(self, e_x, e_y, a, b, e_angle,n):
        angles_circle = numpy.arange(0, 2 * numpy.pi, 2 * numpy.pi/n)
        x = []
        y = []
        for angles in angles_circle:
            or_x = a * math.cos(angles)
            or_y = b * math.sin(angles)
            length_or = math.sqrt(or_x * or_x + or_y * or_y)
            or_theta = math.atan2(or_y, or_x)
            new_theta = or_theta + e_angle/180*numpy.pi
            new_x = e_x + length_or * math.cos(new_theta)
            new_y = e_y + length_or * math.sin(new_theta)
            x.append(new_x)
            y.append(new_y)
        return x, y



    def environment_moving(self):
        self.t += self.timestep
        '''
        if self.t < 3:
            #print(self.t,self.is_traget )
            self.is_traget =1
            self.traget[0] = 300
            self.traget[1] = 150
        else:
            self.is_traget =1
            self.traget[0] = 500
            self.traget[1] = 500 
        '''
        if self.t >10000:
            
            self.pred_pos  = self.get_pos_of_predator( 1)
            #print(self.t ,self.pred_pos )
            self.predator_moving = 1 

        ##print(self.graphlist)
        #for graph in range(len(self.graphlist)):
            #print(self.graphlist[graph].nodes())
            #self.get_long_term_insecurity(graph)
        #print(len(self.fish_group))
        for fish in self.fish_group:
            ##print(fish.graphID)
            fish.update_insecurity_and_position()
            #print(fish.pos,fish.next_pos)
        for fish in self.fish_group:
            fish.pos = fish.next_pos.copy()
            fish.vel = fish.next_vel.copy()
            #print(fish.pos,fish.next_pos)
                    
        
    def get_pos_of_predator(self,statr_time = 50):
        dt = self.t - statr_time
        pred_pos  = self.predator.run(dt = dt) 
        if  pred_pos[0]<0:
            pred_pos[0] = 0
            pred_pos[1] = 0
        return pred_pos

    def init_graph(self, agent):
        G = nx.Graph()
        G.add_node(agent)
        
        agent.graphID = len(self.graphlist) - 1
        
        #print('history_number',range(self.history_number))
        #print('G.history_mean_insecurity',G.history_mean_insecurity.queue)
        G.history_mean_insecurity=queue.Queue(self.history_number) 
        for _ in range(self.history_number):
            G.history_mean_insecurity.put(0)
        #print('G.history_mean_insecurity',G.history_mean_insecurity)
        self.graphlist.append(G)
                     
    def add_node(self, agent1 , agent2):
        G = self.graphlist[agent1.graphID]
        G.add_node(agent2)
    def add_edge(self, agent1 , agent2):
        G = self.graphlist[agent1.graphID]
        agent2.graphID = agent1.graphID
        G.add_edge(agent1,agent2)  

    def get_long_term_insecurity(self,graphID):
        G = self.graphlist[graphID]
        
        sum_insecurity = 0
        for Gi in G:
            if Gi == 0:
                continue           
            sum_insecurity += Gi.short_term_insecurity
        if len(G) == 0:
            self.mean_insecurity = 1
        else:
            self.mean_insecurity = sum_insecurity/len(G)
        #print('self.mean_insecurity',self.mean_insecurity) 
        #print(type(G.history_mean_insecurity))
        #print(G.history_mean_insecurity.queue)
        if G.history_mean_insecurity.empty() is not True:
            G.history_mean_insecurity.get()
            #print('G.history_mean_insecurity',G.history_mean_insecurity) 
            G.history_mean_insecurity.put(self.mean_insecurity)
            #print('G.history_mean_insecurity',G.history_mean_insecurity)
            G.long_term_insecurity = sum(G.history_mean_insecurity.queue)/self.history_number
            #print(G.long_term_insecurity)
            return G.long_term_insecurity

    def get_positions(self):
        positions = []
        all = self.fish_group
        for a in all:
            positions.append(a.pos)
        #print( all[2].pos)
        return numpy.array(positions)
    
    def get_velocity(self):
        vel = []
        all = self.fish_group 
        for a in all:
            vel.append(a.vel / fast_norm(a.vel))                   
        return numpy.array(vel)

    def get_ori_mean_angles(self):
        ori_mean_angles = []
        all = self.fish_group 
        for a in all:
            ori_mean_angles.append(a.ori_mean_angle )                   
        return ori_mean_angles
        
#Environment(num_fish=50, DIM=2, timestep = 0.5)
#env = Environment(num_fish=100, DIM=2, timestep = 0.1)
#env.environment_moving()