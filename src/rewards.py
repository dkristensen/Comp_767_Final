import numpy as np
import state_getter as sg
import sys, os

try:
    sys.path.append(os.path.join(os.path.dirname(__file__),"tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci


class RewardsContainer:
    def __init__(self):
        """
        Create the rewards container for the following rewards
        Rewards Used (Both single timestep and change over adjacent timesteps):
        - Queue length
        - Average Delay
        - Average Delay (Squared)
        - Cumulative Delay
        - Cumulative Delay (Squared)
        - Average Arrived Travel Time
        """
        self.n_rewards = 6
        self.current, self.previous = [0]*self.n_rewards, [0]*self.n_rewards
        # Queue length
        self.queue_length_id = 0
        # Average Delay
        self.avg_delay_id = 1
        # Average Delay (Squared)
        self.avg_delay_sq_id = 2
        # Cumulative Delay
        self.cum_delay_id = 3
        # Cumulative Delay (Squared)
        self.cum_delay_sq_id = 4
        # Average Arrived Travel Time
        self.aatt_id = 5

        self.cars = []
        return

    def update_rewards(self):
        self.take_timestep()
        return
    
    def compute_rewards(self):
        self.compute_queue()
        self.compute_oneshot_delays()
        self.compute_delay()
        self.compute_aatt()
        return 

    def get_all_rewards(self):
        """
        Return a list of all the rewards to compute
        """
        self.update_rewards()
        rewards = []
        rewards += self.current
        delta_rewards = [self.previous[i]-self.current[i] for i in range(len(self.current))]
        rewards += delta_rewards
        return rewards



    def take_timestep(self):
        """
        Move all current reward values into their previous counterparts
        """
        self.previous = [val for val in self.current]
        self.compute_rewards()
        return

    def compute_queue(self):
        total_lane_queues = 0
        lanes = traci.lane.getIDList()
        for lane in lanes:
            lane_queue = traci.lane.getLastStepHaltingNumber(lane)
            total_lane_queues += lane_queue
        self.current[self.queue_length_id] = total_lane_queues
        return 

    def compute_oneshot_delays(self):
        traffic_lights = sg.get_all_traffic_light_ids()
        controlled_lanes = [sg.get_incoming_lanes(tlid) for tlid in traffic_lights]
        
        total_vehicles = 0
        total_delays = 0
        total_delays_sq = 0

        for tl in controlled_lanes:
            for lane in tl:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                delays = [traci.vehicle.getAccumulatedWaitingTime(car) for car in vehicles]
                squared_delays = [delay**2 for delay in delays]
                total_vehicles += len(delays)
                total_delays += sum(delays)
                total_delays_sq += sum(squared_delays)
        
        if(total_vehicles>0):
            self.current[self.avg_delay_id] = total_delays/total_vehicles
            self.current[self.avg_delay_sq_id] = total_delays_sq/total_vehicles

            self.current[self.cum_delay_id] = total_delays
            self.current[self.cum_delay_sq_id] = total_delays_sq
        return


    def compute_delay(self, power=1, average=True):
        traffic_lights = sg.get_all_traffic_light_ids()
        controlled_lanes = [sg.get_incoming_lanes(tlid) for tlid in traffic_lights]
        total_vehicles = 0
        total_delays = 0
        for tl in controlled_lanes:
            for lane in tl:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                delays = [traci.vehicle.getWaitingTime(car)**power for car in vehicles]
                total_vehicles += len(delays)
                total_delays += sum(delays)
        if(total_vehicles>0):
            if(average):
                avg_delay = total_delays/total_vehicles
                if(power is 1):
                    self.current[self.avg_delay_id] = avg_delay
                if(power is 2):
                    self.current[self.avg_delay_sq_id] = avg_delay
            else:
                if(power is 1):
                    self.current[self.cum_delay_id] = total_delays
                if(power is 2):
                    self.current[self.cum_delay_sq_id] = total_delays
        return


    def compute_aatt(self, car_limit = 100):
        newly_arrived = [car for car in traci.vehicle.getIDList() if traci.vehicle.getNextTLS(car) is ()]

        for car in newly_arrived:
            self.cars.insert(0, int(traci.simulation.getTime() - int(traci.vehicle.getParameter(car,"departure_time"))) )
            traci.vehicle.remove(car)
            if(len(self.cars)>car_limit):
                self.cars.pop()
        if(len(self.cars)>0):
            aatt_value = sum(self.cars)/len(self.cars)
            self.current[self.aatt_id] = aatt_value
        return 
