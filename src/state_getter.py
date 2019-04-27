"""
File holding all my scripts for getting the state and reward for the network
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import math
import numpy as np
import time

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__),"tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

# Threshold in m/s for cars satisfying our state space condition
SPEED_THRESHOLD = 5
# Max number of possible light phases
LIGHT_CLASSES = 12
LANE_TYPES = 5
MAX_LANES = 6 
ANGLE_VEHICLE_HOLDER = "angle_getter"
speedThresh = 5
TIME_STEP = 1
WAIT_POWER = 2#2
DECEL_PEN = 1#0.5
LIGHT_PEN = 2#10

def get_all_traffic_light_ids():
    """Returns a list holding the ids for all the traffic lights in the system"""
    # only serves as a wrapper for the TraCI command
    return traci.trafficlight.getIDList()

def get_incoming_lanes(traffic_light_id):
    controlled_links = traci.trafficlight.getControlledLinks(traffic_light_id)
    controlled_incoming = list(set([path[0] for [path] in controlled_links]))
    return controlled_incoming

def get_controlled_lanes(traffic_light_id):
    """Returns a list of the lanes that the traffic light controlls broken
    up by incoming direction
    :param traffic_light_id: the id for the traffic light in the network
    :return: A list holding the lane ids for each incoming direction
    """
    # Get traffic lanes controlled by the light and get rid of duplicates
    controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    controlled_lanes = set(controlled_lanes)
    
    # Set the layout for assigning lanes
    directional_name = ['e_0','e_1','e_2','e_3','e_4','e_5']
    directional_lanes = []

    for i in range(MAX_LANES):
        directional_lanes.append([lane for lane in controlled_lanes if "e_{}".format(i) in lane])

    
    # For each direction, assign the lane that is coming in from 
    # that direction to the corresponding lane array
    # for name in directional_name:
    #     directional_lanes.append([lane for lane in controlled_lanes if name in lane])
    directional_lanes = get_lane_types(traffic_light_id, directional_lanes)

    return directional_lanes

def get_angle(currJunc, otherJunc):
    """
    Gets the angle between the current Junction and another junction,
    relative to the x-axis
    """
    return math.atan2(otherJunc[1]-currJunc[1], otherJunc[0]-currJunc[0])


def get_lane_types(tlid, edges):
    if(len(traci.lane.getParameter(traci.lane.getIDList()[-1], "laneType")) is 0):
        return set_lane_types(tlid,edges)
    for edge in edges:
        for i in range(len(edge)):
            edge[i] = ( edge[i], int(traci.lane.getParameter(edge[i],"laneType")))
    return edges


def set_lane_types(tlid, edges):
    """ Set the lane type for each lane in all edges, for use in setting
    the state space
    :param tlid: the id of the traffic light for which to get the lane types from
    :param edges: the edges that feed into the junction
    """
    # Get current junction and its (x,y) location
    my_x,my_y = traci.junction.getPosition(tlid)
    edge_names = [traci.lane.getEdgeID(edge[0]) for edge in edges if len(edge)>0]
    alternative_edges = [edge for edge in traci.edge.getIDList() if edge not in edge_names and ":" not in edge]
    controlled_links = [datum[0] for datum in traci.trafficlight.getControlledLinks(tlid)]
    for edge in edges:
        if(edge):
            if(edge[0] not in traci.route.getIDList()):
                for fromLane,toLane,_ in controlled_links:
                    if(edge[0] == fromLane):
                        traci.route.add(edge[0],[traci.lane.getEdgeID(fromLane),traci.lane.getEdgeID(toLane)])
                        break
            traci.vehicle.add(vehID= ANGLE_VEHICLE_HOLDER, routeID = edge[0])
            traci.vehicle.moveTo(ANGLE_VEHICLE_HOLDER, edge[0], 5)
            edgeid = traci.lane.getEdgeID(edge[0])
            
            angle = (90-traci.vehicle.getAngle(ANGLE_VEHICLE_HOLDER))%360
            angle_in_rads = angle*math.pi/180
            # print(edgeid,angle_in_rads)
            for i in range(len(edge)):
                lane = edge[i]
                links = traci.lane.getLinks(lane)
                laneClass=None
                left,center,right = 0,0,0
                for link in links:
                    
                    loc = traci.junction.getPosition(link[0].split('e')[0])
                    angle = (angle_in_rads - get_angle((my_x,my_y), loc))% (2*math.pi)


                    if(abs(angle)<=1*math.pi/8):
                        center = 1
                    elif(angle<math.pi):
                        right = 1
                    elif(angle>=math.pi):
                        left = 1
                laneClass = (2*center + 4*right)/(left+center+right)#Dont need to add 0*Left since its 0

                edge[i] = (lane,laneClass)
                traci.lane.setParameter(lane, "laneType", str(laneClass))
            traci.vehicle.remove(ANGLE_VEHICLE_HOLDER)
    return edges


def get_DTSE(n_intersections = 1, n_edges = 4, cell_size = 10):
    MAX_LANE_LENGTH = 0
    ACTION_SPACE = 4

    tls = get_all_traffic_light_ids()
    junction_lanes = [get_incoming_lanes(tl) for tl in tls]
    if(MAX_LANE_LENGTH is 0):
        for jl in junction_lanes:
            for lane in jl:
                length = traci.lane.getLength(lane)
                if(length>MAX_LANE_LENGTH):
                    MAX_LANE_LENGTH = length
    # Shape is (all intersections,
    #            the traffic signals we can have, 
    #            the number of cells in our lanes,
    # and the binary encoding of position and real valued proprotion of speed limit time the lanes in our junctions)
    n_cells = int(MAX_LANE_LENGTH/cell_size)
    DTSE_state = np.zeros(shape=(len(junction_lanes),ACTION_SPACE,len(junction_lanes[0]),2*n_cells))
    for i in range(len(junction_lanes)):
        current_signal = traci.trafficlight.getPhase(tls[i])
        current_signal = current_signal//2 if current_signal<8 else 2
        
        junction = junction_lanes[i]
        for j in range(len(junction)):
            # i is traffic light index
            # j is lane index within traffic light
            lane = junction[j]
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
            lane_length = traci.lane.getLength(lane)
            speed_limit = traci.lane.getMaxSpeed(lane)
            for vehicle in vehicles_on_lane:
                index_along_lane = int((lane_length - traci.vehicle.getLanePosition(vehicle))/cell_size)
                relative_speed = traci.vehicle.getSpeed(vehicle)/speed_limit
                DTSE_state[i,current_signal,j,index_along_lane] += 1
                DTSE_state[i,current_signal,j,index_along_lane+n_cells] += relative_speed
            for k in range(int(MAX_LANE_LENGTH/cell_size)):
                if(DTSE_state[i,current_signal,j,k] > 1):
                    DTSE_state[i,current_signal,j,k+n_cells] /= DTSE_state[i,current_signal,j,k]
                    DTSE_state[i,current_signal,j,k] = 1
            
    return DTSE_state

def set_767_action(action):
    tl = get_all_traffic_light_ids()[0]
    curr = traci.trafficlight.getPhase(tl)
    action *= 2
    if(curr == action):
        traci.trafficlight.setPhase(tl,action)
        traci.trafficlight.setParameter(tl,"next_phase",str(action+1))
        return
    elif(str(action) == traci.trafficlight.getParameter(tl,"next_phase") and (traci.trafficlight.getNextSwitch(tl) - traci.simulation.getTime())>=3):
        traci.trafficlight.setPhase(tl,action)
        return
    else:
        transition_action = compute_transition(curr,action)
        traci.trafficlight.setPhase(tl,int(transition_action))
        traci.trafficlight.setParameter(tl,"next_phase",str(action))
        return
    
def compute_transition(current, selected):
    if(current is 0):
        if(selected is 2):
            return 1
        else:
            return 8
    elif(current is 2):
        if(selected is 0):
            return 0
        else:
            return 3
    elif(current is 4):
        if(selected is 6):
            return 5
        else:
            return 9
    elif(current is 6):
        if(selected is 4):
            return 4
        else:
            return 7
    elif((current%2 == 1) or current == 8):
        return selected
    return selected# (current+1)%10

def set_departure_times():
    if(traci.simulation.getDepartedNumber()>0):
        for departed_vehicle in traci.simulation.getDepartedIDList():
            traci.vehicle.setParameter(departed_vehicle,"departure_time",str(int(traci.simulation.getTime())))
    return
    
def get_network_state():
    """Returns the state for all lights in the network"""
    # Get the ids of all the traffic lights in the network
    network_lights = get_all_traffic_light_ids()
    network_lanes = map(get_controlled_lanes, network_lights)

    # Get the list of all cars in each lane for the traffic lights
    cars_in_lanes = map(get_num_cars_in_lanes, network_lanes)
    # Get the list of all cars in each lane under a threshold for the traffic lights
    cars_under_thresh = map(get_cars_under_threshold_at_light, network_lanes)
    # Get the list of all current phases for the traffic lights
    current_phases = map(get_traffic_light_phase, network_lights)
    # Zip them all together so we can have each entry containing the state space for that light
    state_space = zip(cars_in_lanes,
                        cars_under_thresh,
                        current_phases)
    return state_space

def get_flattened_states():
    """Returns the state of the network in a flattened form"""
    states = get_network_state()
    flattened_state = [[]]*len(states)
    count = 0
    for state in states:
        for data in state:
            if(type(data[0]) is list):
                data = [value for subarray in data for value in subarray]
            flattened_state[count].extend(data)
        count+=1
    return flattened_state[0]

def get_num_cars_in_lanes(lanes):
    """Returns the number of cars in each lane at the light 
    :param lanes: the lanes of the traffic light to get the cars from
    :return: the number of cars moving less than our threshold in each lane
    """
    num_cars = []
    for direction_index in range(len(lanes)):
        num_cars.append([0]*LANE_TYPES)
        direction = lanes[direction_index]        
        cars = [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id,_ in direction]
        for i in range(len(direction)):
            num_cars[direction_index][direction[i][1]] += cars[i]
    num_cars = normalize_list(num_cars)
    return num_cars

def get_cars_under_threshold_at_light(lanes, thresh=SPEED_THRESHOLD):
    """Returns the number of cars in each lane moving less than some speed at the light
    :param lanes: the lanes of the traffic light to get the cars from
    :return: the number of cars moving less than our threshold in each lane
    """
    num_under_thresh = []
    for direction_index in range(len(lanes)):
        direction = lanes[direction_index]
        current_direction = [0]*LANE_TYPES
        
        for lane,lane_type in direction:
            # Get the vehicles on each lane and their speeds
            vh_ids = traci.lane.getLastStepVehicleIDs(lane)
            vh_speeds = map(traci.vehicle.getSpeed, vh_ids)
            # Get the number of cars that are going under the threshold declared at the top of the file
            current_direction[lane_type] = (len([car for car in vh_speeds if car<=thresh]))
        num_under_thresh.append(current_direction)
    num_under_thresh = normalize_list(num_under_thresh)
    return num_under_thresh

def normalize_list(car_list):
    total = sum(sum(sub_list) for sub_list in car_list)
    if(total > 0):
        for i in range(len(car_list)):
            car_list[i] = [float(value)/total for value in car_list[i]]#[math.log(value+1) for value in car_list[i]]#
    return car_list

def get_traffic_light_phase(trafficLightID):
    """Returns the phase of the traffic light in the form of a one hot vector
    :param trafficLightID: The id of the traffic light to get the phase of
    :return: A one hot list holding the current traffic state
    """
    phase = traci.trafficlight.getPhase(trafficLightID)
    phase_vector = [0]*LIGHT_CLASSES
    unchanged = traci.trafficlight.getParameter(trafficLightID,"time_unchanged")
    if(unchanged == ""):
        unchanged = 1
        traci.trafficlight.setParameter(trafficLightID,"time_unchanged","1")
    else: unchanged = int(unchanged)
    phase_vector[phase] = unchanged
    return phase_vector


def set_traffic_light_phase(traffic_phases):
    """Sets the traffic phases passed to it by setting each light in the order
    provided by traci.trafficlight.getIDList()
    :param traffic_phases: the index of each of the traffic signals in their
    respective traffic phase definitions
    :return: Any penalty associated with setting the lights to the values passed
    """
    tlsids = get_all_traffic_light_ids()
    light_penalty = 0
    for i in range(len(tlsids)):
        light_penalty += set_traffic_phase(tlsids[i], traffic_phases[i])
    return light_penalty

def set_traffic_phase(tlid, tlphase):
    light = traci.trafficlight.getCompleteRedYellowGreenDefinition(tlid)
    phases = str(light[0]).count("phaseDef")
    current = traci.trafficlight.getParameter(tlid,"time_unchanged")
    if(current == ""): current = 0
    else: current = int(current)
    if(tlphase>(phases-1)):
        if(traci.trafficlight.getPhase(tlid) == (phases-1)):
            traci.trafficlight.setParameter(tlid,"time_unchanged", str(current+1))
        else:
            traci.trafficlight.setParameter(tlid,"time_unchanged", str(1))
        traci.trafficlight.setPhase(tlid,(phases-1))
        return LIGHT_PEN
    else:
        if(traci.trafficlight.getPhase(tlid) == tlphase):
            traci.trafficlight.setParameter(tlid,"time_unchanged", str(current+1))
        else:
            traci.trafficlight.setParameter(tlid,"time_unchanged", str(1))
        traci.trafficlight.setPhase(tlid,tlphase)
        return 0

def update_car(car_list_entry):
    # order of entry : car,lane,time,speed
    if(traci.vehicle.getLaneID(car_list_entry[0]) == car_list_entry[1]):
        if(traci.vehicle.getSpeed(car_list_entry[0]) <= speedThresh):
            return [car_list_entry[0],car_list_entry[1],car_list_entry[2]+1,traci.vehicle.getSpeed(car_list_entry[0])]
    else:
        return [car_list_entry[0],traci.vehicle.getLaneID(car_list_entry[0]),0,traci.vehicle.getSpeed(car_list_entry[0])]
    return [car_list_entry[0],car_list_entry[1],car_list_entry[2],traci.vehicle.getSpeed(car_list_entry[0])]



def get_cars_list(old_car_list):
    """
    """
    
    arrived_cars = traci.simulation.getArrivedIDList()
    # remove all cars that are no longer in the network
    old_car_list = [datum for datum in old_car_list if datum[0] not in arrived_cars]

    old_car_list = map(update_car,old_car_list)
    for car,lane,time,speed in old_car_list: traci.vehicle.setParameter(car,"edgeTime",str(time))

    departed_cars = traci.simulation.getDepartedIDList()
    departed_cars = [car for car in departed_cars if car != ANGLE_VEHICLE_HOLDER]

    departed_list = [[car,traci.vehicle.getLaneID(car),0,traci.vehicle.getSpeed(car)] for car in departed_cars]
    new_car_list = old_car_list + departed_list

    return new_car_list


def get_car_list_waits(car_list):
    """Returns the average squared waits of all the vehicles in car_list
    :param car_list: The list of cars in the network
    :return: float holding average squared wait
    """
    wait = 0.0
    for car,lane,time,speed in car_list:
        wait+= time**(WAIT_POWER)
    if(wait == 0):
        return wait
    return math.log(wait)+1


def check_cars_decel(cars_list):        
    loaded_cars = traci.vehicle.getIDList()
    penalty = 0
    for car,_,_,old_speed in cars_list:
        if(car not in loaded_cars):
            continue
        elif((old_speed-traci.vehicle.getSpeed(car))/TIME_STEP > traci.vehicle.getEmergencyDecel(car)):
            penalty += DECEL_PEN
    return penalty





def eval_update_car(car_list_entry):
    # order of entry : car,lane,time,speed
    if(traci.vehicle.getLaneID(car_list_entry[0]) == car_list_entry[1]):
        if(traci.vehicle.getSpeed(car_list_entry[0]) <= speedThresh):
            return [car_list_entry[0],car_list_entry[1],car_list_entry[2]+1,traci.vehicle.getSpeed(car_list_entry[0])]
    else:
        value = traci.vehicle.getParameter(car_list_entry[0],"squared_edge_wait_time")
        if(value != ""): value = int(value)
        else:            value = 0
        # print(traci.vehicle.getParameter(car_list_entry[0],"squared_edge_wait_time"),type(traci.vehicle.getParameter(car_list_entry[0],"squared_edge_wait_time")))
        traci.vehicle.setParameter(car_list_entry[0],"squared_edge_wait_time",str(value+(car_list_entry[2]**WAIT_POWER)))
        return [car_list_entry[0],traci.vehicle.getLaneID(car_list_entry[0]),0,traci.vehicle.getSpeed(car_list_entry[0])]
    return [car_list_entry[0],car_list_entry[1],car_list_entry[2],traci.vehicle.getSpeed(car_list_entry[0])]



def eval_get_cars_list(old_car_list):
    """
    """
    
    arrived_cars = traci.simulation.getArrivedIDList()
    arrived_waits,arrived_num = 0.0,0.0

    # remove all cars that are no longer in the network
    old_car_list = [datum for datum in old_car_list if datum[0] not in arrived_cars]
    for car,lane,time,speed in old_car_list:
        i = old_car_list.index([car,lane,time,speed])
        if(traci.vehicle.getRoute(car)[-1] == traci.lane.getEdgeID(lane)):
            arrived_num += 1
            arrived_waits+=int(traci.vehicle.getParameter(car,"squared_edge_wait_time"))
            traci.vehicle.remove(car)
            old_car_list.pop(i)
            # print(int(traci.vehicle.getParameter(car,"squared_edge_wait_time")))
    

    old_car_list = map(eval_update_car,old_car_list)

    departed_cars = traci.simulation.getDepartedIDList()
    departed_cars = [car for car in departed_cars if car != ANGLE_VEHICLE_HOLDER]

    departed_list = [[car,traci.vehicle.getLaneID(car),0,traci.vehicle.getSpeed(car)] for car in departed_cars]
    new_car_list = old_car_list + departed_list

    return new_car_list, arrived_waits, arrived_num