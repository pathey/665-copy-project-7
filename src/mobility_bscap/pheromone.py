#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 2024

@author: shreyasdevaraju
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Modified form (Lambda) home/shivamgarg/Desktop/BSCAPscheme_toadd_Routing/TOPO-CONTROL-Sep28/BS-CAP3-returnnxthop-20ms-3TARGETS-DroppingNodes-PIPE-TC/

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from collections import deque
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, ParamSpec, TypeVar, cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
# from scipy.signal import convolve2d
from numpy.typing import NDArray
from scipy.ndimage import convolve as convolveim
import time
import random
import networkx as nx
# import q_agent_V2 as ql # FOR DQN only (commented)
# import globals  # contains global variables (commented)
from numba import jit
import math

import simpy
from simpy import Process
from simpy import Timeout

arr = np.asarray

#ROUTING
# import route_selection
#Generak parameters and route details               #SHREY*
import parameters

#RouteFunctions.py contains the network update     #SHREY*
import RouteFunctions as rf

from line_profiler import profile

from node import Node

P = ParamSpec("P")
R = TypeVar("R")

try:
    @profile
    def check_for_line_profiler() -> None:
        pass
except:
    def profile(f: Callable[P, R]) -> Callable[P, R]:
        return f


class UAVStats(object):
    """ Defines metrics used for results and analysis """
    @profile
    def __init__(self, nAgents: int, nBaseStations: int, xgv: NDArray[np.floating], ygv: NDArray[np.floating], sim_time: int):  # BSedit
        self.coverage: list[float] = [0]
        self.cellscoverage_per_100s: list[float] = []
        self.visited: NDArray[np.int8] = np.zeros((len(xgv)+2, len(ygv)+2), dtype=np.int8)  # __grid;
        self.frequency: NDArray[np.int32] = np.zeros((len(xgv)+2, len(ygv)+2), dtype=np.int32)  # __grid;
        self.fairness: list[float | np.floating] = []
        self.largest_subgraph: list[float] = []
        self.no_connected_comp: list[int] = []
        self.freq_subgraph_sizes: NDArray[np.floating] = np.zeros(nAgents-1)
        self.avg_deg_conn: list[np.floating] = []
        self.is_biconnected_Gaint_Comp: list[bool] = []
        self.cell_visted_times: NDArray[np.int32] = np.zeros((len(xgv)+2, len(ygv)+2, 4000), dtype=np.int32)

        self.total_time_connected_to_BS: NDArray[np.int8] = np.zeros((nAgents, nBaseStations, sim_time+1), dtype=np.int8)
        
        self.runtime: list[float] = []

        #UAV position
        self.UAV_positions = np.zeros((sim_time+1, nAgents+nBaseStations, 2))  

        #UAV energies
        self.UAV_energies = np.zeros((sim_time+1, nAgents+nBaseStations))  

        # Target and Target UAV information

        # midcoverage
        self.visited_after_genstart: NDArray[np.int8] = np.zeros((len(xgv)+2, len(ygv)+2), dtype=np.int8)  # __grid;
        self.frequency_after_genstart: NDArray[np.int32] = np.zeros((len(xgv)+2, len(ygv)+2), dtype=np.int32)  # __grid;
        self.coverage_after_genstart: list[float] = [0]
        self.fairness_after_genstart: list[float | np.floating] = []
        self.largest_subgraph_after_genstart: list[float] = []
        self.no_connected_comp_after_genstart: list[int] = []
        self.freq_subgraph_sizes_after_genstart: NDArray[np.floating] = np.zeros(nAgents-1, dtype=np.float64)
        self.avg_deg_conn_after_genstart: list[np.floating] = []
        self.is_biconnected_Gaint_Comp_after_genstart: list[bool] = []
        self.total_time_connected_to_BS_after_genstart: NDArray[np.int8] = np.zeros((nAgents, nBaseStations, sim_time+1), dtype=np.int8)
        

        #hist values of visitation freq
        self.vfhist_0=[]
        self.vfhist_1k=[]

        
        self.percentage_coverage_in_episode = 0.0

        # self.skip=True # flag for skipping a particular run results

        pass


class UAV_MultipleTargets(object):
    """Defines Multiple Targets, their location and acquisition status"""
    @profile
    def __init__(self, target_locations):
        self.number_of_targets =len(target_locations)

        self.location_xy = np.array(target_locations)  #[[2000, 3000], [4000, 3000]]    

        self.service_status = np.array( [False for x in range(self.number_of_targets)] ) #False 
        self.TargetUAV_assigned = np.array( [None for x in range(self.number_of_targets)] ) #None
        self.TargetUAV_assigned_Time = np.array( [None for x in range(self.number_of_targets)] )


class uav_swarm(object):
    """Multi-agent UAV swarm simulation using pheromone tracking"""
    @profile
    def __init__(self, env, **kwargs):
        self.env: simpy.Environment = env    #SHREY*
        self.stats = None

        self.evaporation_rate = kwargs.get('evaporation_rate',    .1)
        self.diffusion_rate = kwargs.get('diffusion_rate',      .1)

        self.time_step = kwargs.get('time_step',      1)           # time step resolution dt
        self.stats_interval: int = kwargs.get('stats_interval',  100)   # stats log interval
        self.hello_period = kwargs.get('hello_period',   2)        # hello message interval
        # self.sim_time     = kwargs.get('sim_time',     400);  # made sim parameter
        self.nAgents: int = kwargs.get('nAgents',       20)          # number of UAVs 
        self.nBaseStations: int = kwargs.get('nBaseStations', 1)
        self.nHistory: int = kwargs.get('nHistory',      40)         # previous time step location history log    
        self.use_pheromone = kwargs.get('use_pheromone', True)  # pheromone model flag 
        self.use_connect = kwargs.get('use_connect', True)      # pheromone model + connectivity flag 
        self.min_degree = kwargs.get('min_degree',     2)  # not used
        self.map_size: float = kwargs.get('map_size',    4000)         # map size
        self.map_resolution: float = kwargs.get('map_resolution', 100)  # cell size =100x100m
        self.transmission_range: float = kwargs.get('transmission_range', 1000) # TODO: Convert into array which has actual ranges for each node
        self.fwd_scheme: int = kwargs.get('fwd_scheme',     5)  # forward 5 directions for next waypoint selection
        self.hop_dist = kwargs.get('hop_dist',       2)    # number of cells away for next waypoint
        self.collision_avoidance = kwargs.get('collision_avoidance', True)
        self.turn_buffer = kwargs.get('turn_buffer',  40)             # turn radius buffer needed at boundaries
        self.collision_buffer = kwargs.get('collision_buffer', 40)    # collision buffer
        self.uav_airspace = self.collision_buffer   # individual UAV air space
        self.waypoint_radius = kwargs.get('waypoint_radius', 50)  # radius around waypoint that is consider to check if UAV reach waypoint
        
        self.decision_bs_connections = kwargs.get('decision_bs_connections', self.nBaseStations)
        
        # connectivity_scheme =1 1hop , 2 2hop scheme
        self.connectivity_scheme = kwargs.get('connectivity_scheme',     1)  # to turn on connectivity condition; not used

        # resolution / locations for grids # number of cells in x and y axis
        self.xgv: NDArray[np.floating]
        self.ygv: NDArray[np.floating]
        # alternate grid? (unused)
        self.xvv: NDArray[np.floating]
        self.yvv: NDArray[np.floating]

        self.colors = np.array([[0.859, 0.251, 0.251], [0.969, 0.408, 0.408], [1.0, 0.588, 0.588],   # used in plotting (not used)
                                [0.737, 0.122, 0.122], [0.585, 0.035, 0.035], [1.0, 0.0, 0.0],
                                [0.988, 0.102, 0.102], [1.0, 0.266, 0.266], [0.784, 0.0, 0.0]])
        
        # start positions of UAVs
        self.init_posX = kwargs.get('init_posX',    np.array((self.map_size/2) + (-1+2*np.random.rand(self.nAgents+self.nBaseStations))*200))
        self.init_posY = kwargs.get('init_posY',    np.array(300 + (-1+2*np.random.rand(self.nAgents+self.nBaseStations))*200))

        #INTIALIZE Base Station ID and its positions 
        # the last Agent id is BS; if 31 UAVS then 30 is the BS_id  (# "BSedit" is added lines to incorporate BS connectivity)
        assert self.nBaseStations <= 4, "Only Up to 4 Base Stations are allowed"

        CHOSEN_BS_SCHEME = parameters.MOBILITY.BS_SCHEME
        assert self.nBaseStations <= len(CHOSEN_BS_SCHEME), "Too many Base Stations for the chosen BS scheme"
        
        self.bs_iDs: list[int] = [self.nAgents+i for i in range(self.nBaseStations)]
        self.bs_positions = []
        for pos_idx in range(self.nBaseStations):
            self.bs_positions.append(CHOSEN_BS_SCHEME[pos_idx])
            
            
        for i, bsID in enumerate(self.bs_iDs):
            self.init_posX[bsID] = self.bs_positions[i][0]   #BSedit BS Location
            self.init_posY[bsID] = self.bs_positions[i][1]   #BSedit

        self.init_map()
        self.init_agents()

        #default target location
        self.target_location = kwargs.get('target_location', [3000,3000])
        self.init_targets(self.target_location) # create targets object using target locations

        self.alpha_type = kwargs.get('alpha_type', 0)
        self.power = kwargs.get('power', 1)

        #Drop Percentage of nodes
        self.drop_percent = kwargs.get('drop_percent', 0)

        # Pipe_neighbor_density_threshold
        self.Pipe_neighbor_density_threshold = kwargs.get('Pipe_neighbor_density_threshold', 2)

    @profile
    def init_targets(self, target_location):
        """Initialize Targets to Track"""
        self.multipletarget = UAV_MultipleTargets(target_location)

    @profile
    def init_map(self):
        """Re-initialize grids and pheromone maps"""
        self.xgv = np.linspace(0, self.map_size, int(
            self.map_size/self.map_resolution))  # grid size 1 unit
        self.ygv = np.linspace(0, self.map_size, int(
            self.map_size/self.map_resolution))  # grid size 1 unit
        # ATI: removed unused code to use a different grid size for visitation counts

        self.pheromone = self.__grid()   # pheromone map

        self.node_pheromone_map = np.tile(
            self.__grid()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations))
        self.node_pheromone_Attract = np.tile(
            self.__grid()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations))
        self.node_pheromone_Repel = np.tile(
            self.__grid()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations))
        self.node_pheromone_Tracking = np.tile(
            self.__grid()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations))
        
        self.node_pheromone_PIPE = np.tile(
            self.__grid()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations))        

        #################################################################################
    @profile
    def init_agents(self):
        """Initialize the state of the controllers & robots"""
        self.Acontroller: NDArray[np.float64] = np.zeros((14, self.nAgents + self.nBaseStations), dtype=np.float64)                      # controller & robot states (TODO: fix)
        self.Arobot: NDArray[np.float64] = np.zeros((5, self.nAgents + self.nBaseStations), dtype=np.float64)                            # physical robot states

        self.Arobot_prev_cell: NDArray[np.int32] = np.zeros((2, self.nAgents + self.nBaseStations), dtype=np.int32)
        # save position history for drawing
        self.Arobot_history: NDArray[np.float64] = np.zeros((self.nHistory, 3, self.nAgents + self.nBaseStations), dtype=np.float64) + np.nan

        self.prev_state = np.zeros((self.nAgents + self.nBaseStations, 2*self.fwd_scheme), dtype=np.ndarray)
        self.prev_action = np.zeros((self.nAgents + self.nBaseStations), dtype=int)

        # Create the agents and initialize
        for Id in range(self.nAgents + self.nBaseStations):
            # UAV positions and heading
            self.Acontroller[0, Id] = self.init_posX[Id] # x coordinate
            self.Acontroller[1, Id] = self.init_posY[Id] # y coordinate    
            self.Acontroller[2, Id] = 0.0 # (not used for now) z coordinate
            self.Acontroller[3, Id] = 0  # heading (theta)   /360 * rand
            self.Acontroller[4, Id] = 0  # not used (phi)
            # Commands
            self.Acontroller[5, Id] = 0  # velocity
            self.Acontroller[6, Id] = 0  # mu
            # Memory
            self.Acontroller[7, Id] = Id   # id
            self.Acontroller[8, Id] = 1    # state
            self.Acontroller[9, Id] = 0    # neighbor.id
            self.Acontroller[10, Id] = np.inf  # neighbor.distance
            # Waypoint related information
            self.Acontroller[11, Id] = 0.0   # target x postion (waypoint)
            self.Acontroller[12, Id] = 0.0   # target y postion (waypoint)
            self.Acontroller[13, Id] = 0   # steps (waypoint)

            if Id in self.bs_iDs:
                self.Acontroller[11, Id] = self.bs_positions[self.bs_iDs.index(Id)][0]
                self.Acontroller[12, Id] = self.bs_positions[self.bs_iDs.index(Id)][1]

            # Physical Robot state used in move()
            self.Arobot[0:3, Id] = np.copy(
                self.Acontroller[0:3, Id])  # position xyz
            self.Arobot[3, Id] = 0                          # heading theta
            self.Arobot[4, Id] = 0                          # phi

            self.Arobot_prev_cell[:, Id] = [-1, -1]   # previous cell

        # Initialise communications
        self.channel = self.initChannel(self.nAgents + self.nBaseStations)

    # helper for creating properly-sized grids
    @profile
    def __grid(self): return np.zeros((len(self.xgv)+2, len(self.ygv)+2))

    # helper for creating properly-sized grids
    @profile
    def __grid_ones(self): return np.ones((len(self.xgv)+2, len(self.ygv)+2))    

    # UAV bounds:                      min speed   max speed  min turn   max turn
    # SHREY: changed min speed : 10*dt to 20*dt
    @profile
    def __boundv(self, dt): return arr([parameters.NODE_SPEED*dt,     parameters.NODE_SPEED*dt,   -3*dt,   3*dt])

    @profile
    def __initUAVfigure(self): plt.figure(1); ax = plt.subplot(111, projection='3d'); plt.grid(True); return ax

    def sim_start_3d(self, nodes: list[Node], sim_time: int, episode: int,  drawUAVs: bool = False, drawMap: bool = False, stats: None | UAVStats = None, plotInterval: int = 10) -> Generator[Timeout | Process, None, None | UAVStats]:
        """Run simulation of UAVs with pheromone-based directions"""

        starttime = time.time()
        # initialize statistics object if not passed in
        if stats is None:
            self.stats = UAVStats(self.nAgents, self.nBaseStations, self.xgv,
                             self.ygv, sim_time)  # BSedit
        else:
            self.stats = stats            

        # initiallize evp and diff rates
        evaporation_rate = self.evaporation_rate
        diffusion_rate = self.diffusion_rate

        # coverage metrics
        # self.__grid(); #SD
        temp_100s: NDArray[np.floating] = np.zeros((len(self.xgv)+2, len(self.ygv)+2))

        connMat100: NDArray[np.bool] = np.zeros((self.nAgents + self.nBaseStations, self.nAgents + self.nBaseStations), dtype=bool)
        connMat400: NDArray[np.bool] = np.zeros((self.nAgents + self.nBaseStations, self.nAgents + self.nBaseStations), dtype=bool)



        # ------------------------------------------------------------------------------------
        # Plotting flags -ON/OFF
        drawplots = drawUAVs
        drawAirspace = False           # SHREY draw circle range and repullision radius
        drawUAVflight = drawUAVs
        drawUAVconnectivity = drawUAVs
        drawPheromonemap = drawMap  # true or false for image of pheromone map
        plot_interval = plotInterval
        # ax,axx = None,None     # pre-plotting, no axes exist yet
        ax, axx = None, (70, 5)    # initial elevation & azimuth
        # ------------------------------------------------------------------------------------

        # % Open new figure window
        # if (drawplots):
        #     plt.figure(1)

        # Initialise Time
        # t = 0               #SHREY* COMMENTED
        dt = parameters.LOCATION_UPDATE_INTERVAL/1e9 # dt = self.time_step #SHREY* before change

        # initial taget location
        print("Multiple Targets located at ", self.multipletarget.location_xy)

        # UAV positions at t =0
        self.stats.UAV_positions[0,:,:]= self.Arobot[0:2, :].T



        # #--------------------------------------------------------------
        # ##Energy Intialization of UAVs
        # #--------------------------------------------------------------
        """
          Set in parameters.py
        """

        #---------------------------------------------------------------------------------------------------------------------
        # UAV Energies at t=0
        self.stats.UAV_energies[0,:] = parameters.MOBILITY.NODE_ENERGY
        #---------------------------------------------------------------

        # Call routing function at t=0
        # hops=[]
        # AODV_paths=[]
        # AODV_data_rates=[]
        # #used in broken delay part
        # last_paths=[]
        # last_path_costs=[]
        
        prev_paths = []
        prev_path_costs = []
        # broken_delay =[]
        # breakdelay_counter=0
        # Efail=0
        # Eth_below=0
        # route_length_array=np.zeros( (number_of_nodes, sim_time+1)) #total_simulation_time) ) 


        if not parameters.MOBILE_SCENARIO and not parameters.FORCED_ROUTES:
            for tt, (target_x, target_y) in enumerate(self.multipletarget.location_xy):
                distances = []
                for node in nodes:
                    node_x, node_y = node.latitude, node.longitude

                    distances.append((np.hypot(node_x - target_x, node_y - target_y), rf.getNodeIdx(node.name)))
                distances.sort()
                
                potential_dist, potential_id = distances[0]
                    
                if potential_dist >= self.transmission_range-1:
                    print(tt, f"({target_x}, {target_y})", "Dist:", potential_dist, "ID:", potential_id)
                    raise ValueError("No node within transmission range of target?")
                
                if not self.multipletarget.service_status[tt]:
                    self.multipletarget.TargetUAV_assigned[tt] = potential_id
                    self.multipletarget.TargetUAV_assigned_Time[tt] = 0
                    self.multipletarget.service_status[tt] = True

                    # Update controller information
                    controller = self.Acontroller[:, potential_id]
                    controller[11:13] = target_x, target_y
                    controller[8] = 3  # flag set to hover above the target
                    
                    print(f"Target-{tt} found by UAV-{potential_id} at ({target_x}, {target_y}), at time 0")

        # Main simulation loop
        while True:                                         #SHREY*
            t = int(self.env.now / 1e9)
            tidx = t
            # print("MOBILILTY TIME = ", t)

            #--------------------------------------------------------------
            ## Check if Source Nodes or Target UAV 
            #---------------------------------------------------------------------------------------------------------------------
            source_node=[]
            for uavID in self.multipletarget.TargetUAV_assigned:
                if uavID is not None:
                    source_node.append(uavID)

            
            #--------------------------------------------------------------
            ## Node Energy depletion at each time step
            #--------------------------------------------------------------
            # Decrease energy
            parameters.MOBILITY.NODE_ENERGY[0:-self.nBaseStations] -= parameters.MOBILITY.ENERGY_DEPLETION_RATE  # Linear decrease in energy for 1000s
            # parameters.MOBILITY.NODE_ENERGY0:-self.nBaseStations] = parameters.MOBILITY.NODE_ENERGY[0:-self.nBaseStations] - parameters.MOBILITY.energy_depletion_rate # energy_depletion_rate=0.1 # linear drease in energy for 1000s

            # Ensure source and BS node energy remains at maximum
            parameters.MOBILITY.NODE_ENERGY[source_node] = parameters.MOBILITY.MAX_INITIAL_ENERGY  
            # parameters.MOBILITY.NODE_ENERGY[self.bs_iD] = parameters.MOBILITY.Max_initial_node_Energy  

            # Identify failed nodes
            failed_nodes = parameters.MOBILITY.NODE_ENERGY <= parameters.MOBILITY.ENERGY_NODE_FAILURE_VALUE

            if np.any(failed_nodes):
                # Mark failed nodes with failure flag
                parameters.MOBILITY.NODE_ENERGY[failed_nodes] = -9999  # -9999 is used as Failed node flag

                # Drop failed nodes: force move to location [-9999, -9999]
                self.Acontroller[0:2, failed_nodes] = -9999
                self.Acontroller[11:13, failed_nodes] = -9999
                self.Arobot[0:2, failed_nodes] = -9999

            #--------------------------------------------------------------
            ## Energy 
            # UAV Energies at time step t
            self.stats.UAV_energies[t, :] = parameters.MOBILITY.NODE_ENERGY
            #--------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------------


            if parameters.MOBILE_SCENARIO:
                node_pheromone_map_Tminus1 = np.copy(self.node_pheromone_map)
                for uavID in range(self.nAgents):  # BSedit
                    # Get simulation values for the node
                    controller = self.Acontroller[:, uavID]
                    robot = self.Arobot[:, uavID]
                    # Prev_state, action of  each UAV for Q-learning
                    prev_state = self.prev_state[uavID]
                    prev_action = self.prev_action[uavID]

                    # Take measurement
                    # Controller
                    # Receive messages from other agents
                    msgs = self.simReceive(self.channel)
                    if (t > 2):
                        msgs[:, uavID] = [controller[0], controller[1], controller[2], controller[3],
                                        controller[4], controller[5], controller[6], uavID, controller[11], controller[12]]
                        for i in range(len(self.bs_iDs)):
                            msgs[:,self.bs_iDs[i]] = [ self.bs_positions[i][0], self.bs_positions[i][1],0,0,0,0,0,self.bs_iDs[i], self.bs_positions[i][0], self.bs_positions[i][1] ]  #BSedit

                    # Decide where to move next
                    prev_state, prev_action = self.decide_3d(uavID, controller, msgs, dt, prev_state, prev_action, t)

                    if prev_state is not None:  # if prev_state not None, i.e. its returned a new state to be stored to memory
                        # Store each uav's previous state
                        self.prev_state[uavID] = prev_state
                        self.prev_action[uavID] = prev_action

                    # if controller[8] == 3:
                    #     robot[0:2] = [controller[11], controller[12]] # Fix TARGET UAV POSITION on TOP of target; NO REPEL ADDED AT target location
                    # else:

                    # Update position estimates
                    k = controller[5] * controller[6]
                    controller[3] += (k + 2*k + 2*k + k)*dt/6
                    controller[3] = controller[3] % 360

                    # Physical Robot
                    # store in history queue
                    self.Arobot_history[tidx % self.nHistory, :, uavID] = robot[0:3]

                    # % Move the robot
                    robot = self.move(robot, controller[5], controller[6], dt)

                    # Fix TARGET UAV POSITION on TOP of target; REPEL ADDED AT target location
                    if controller[8] == 3:
                        robot[0:2] = [controller[11], controller[12]] 



                    # -----------SEARCH MODE ( using PHEROMONE ) ----------------------------------------------------------------------------
                    
                    pher_cell = tuple(np.ceil(robot[0:2] / self.map_resolution).astype(int))
                    # Shrey:
                    if (self.use_pheromone):
                        # (curr_cellP ~= robot.prev_cell) condition allows for only one deposit per visit by a uav.
                        # 62-2=60=>range between [1,60]
                        if (np.all(arr(pher_cell) >= 1) and np.all(arr(pher_cell) <= (self.pheromone.shape[1] - 2))):
                            if np.any(pher_cell != self.Arobot_prev_cell[:, uavID]):
                                self.Arobot_prev_cell[:, uavID] = pher_cell

            # 12                        self.node_pheromone_Repel[pher_cell+(uavID,)] = 1;   #node_pheromone_Repel(pher_x,pher_y,uavID) + 10 ; %shrey: %NEW increament pheromone by 1

                                # increment counters for output calculation
                                i = pher_cell[0]
                                j = pher_cell[1]

                                self.node_pheromone_Repel[i, j, np.full(9, uavID, dtype=int)] = 1.0   # Repel Pherone =1 is deposited in cell of pheromone map
                                self.stats.visited[i, j] = 1   # log cell visit

                                self.stats.cell_visted_times[i, j, self.stats.frequency[i, j].astype(int)] = t  # log cell visit time

                                self.stats.frequency[i, j] += 1 # log cell visit frequency
                                temp_100s[i, j] = 1

                                # log cell visits after 1000s # After generation start time
                                if t >= parameters.PKT_GENERATION_START_SEC:
                                    self.stats.visited_after_genstart[i, j] = 1
                                    self.stats.frequency_after_genstart[i, j] += 1

                            else:
                                # node_pheromone_Repel(pher_x,pher_y,uavID) /((1-evaporation_rate) * (1-diffusion_rate))
                                self.node_pheromone_Repel[pher_cell + (uavID,)] = 1

                    else:
                        # SHREY:
                        if (np.all(pher_cell >= 0) and np.all(pher_cell < len(self.stats.visited))):
                            # ATI: what is this doing?
                            raise NameError('one count per visit by a uav')

                    # Controller
                    # Retrieve noisy location from GPS;  Shrey: removed gps noise
                    # controller[0:2] = self.gps(robot)[0:2]
                    controller[0:2] = robot[0:2]

                    # Send location to other agents
                    msg = [controller[0], controller[1], controller[2], controller[3], controller[4],
                        controller[5], controller[6], uavID, controller[11], controller[12]]
                    # self.simTransmit(self.channel, uavID, msg);
                    if (t % self.hello_period == 0):
                        self.simTransmit(self.channel, uavID, msg)
                        
                        for i in range(self.nBaseStations):            
                            self.simTransmit(self.channel, self.bs_iDs[i], [
                                        self.bs_positions[i][0], self.bs_positions[i][1], 0, 0, 0, 0, 0, self.bs_iDs[i], self.bs_positions[i][0], self.bs_positions[i][1] ])  #BSedit

                    # Store values
                    self.Acontroller[:, uavID] = controller
                    self.Arobot[:, uavID] = robot

                # ------------PHEROMONE COMPUTATION $ ITS PERIODIC DISTRIBUTION--------------------------------------
                if (self.use_pheromone):
                    # Check connectivity and merge pheromone map of connected UAV neighbors, and this update is done every hello_period=4s,2s (virtual hello packet)
                    if ((t % self.hello_period) == 0):
                        # check connected neighbors
                        connMat = self.connectivity(self.Arobot[0:2, :], self.transmission_range)
                        for uavID in range(self.nAgents + self.nBaseStations):
                            conn_neighbors = np.flatnonzero(connMat[:, uavID])    # get neighbors of uavID

                            # function merges pheromone map of connected UAV neighbors
                            # Merges self.node_pheromone_Repel maps of all neighbor nodes
                            self.merge_pheromone_map(uavID, conn_neighbors)


                    
                    ## FOR REPEL:

                    #  Difussion and evaporation for every node_pheromone_Repel,
                    h = np.ones((3, 3))
                    h[1, 1] = 0
                    convolved_prev_map = convolveim(node_pheromone_map_Tminus1, h[:,:,np.newaxis], mode='constant')
                    
                    for uavID in range(self.nAgents + self.nBaseStations):
                        
                        # ATI faster
                        self.node_pheromone_Repel[:, :, uavID] = ((1-evaporation_rate) * ((1 - diffusion_rate)*self.node_pheromone_Repel[:, :, uavID] + (
                            diffusion_rate/8)*convolved_prev_map[:,:,uavID]))  # %shrey  % NEW(1-evaporation_rate)

                        # To maintain repel value = 1 of cell which UAV is in after difussion, evporation lowers it
                        i = self.Arobot_prev_cell[0, uavID]
                        j = self.Arobot_prev_cell[1, uavID]
                        self.node_pheromone_Repel[i, j, (np.full(9, uavID, dtype=int))] = 1.0  
                    
                    self.node_pheromone_map[:, :, :] = self.node_pheromone_Repel[:, :, :] ## ADDED


                    ##------------------------------------------------------------------------------------------------------
                    ##---START: USE of ATTRACT PHEROMONE - (FOR TOPOLOGY CONTROL - TC_PIPE) ---
                    ##------------------------------------------------------------------------------------------------------
                    if parameters.MOBILITY.TC_PIPE_mobility_ON: 

                        # #REST attract pheromone maps ## ADDED
                        self.node_pheromone_Attract = np.tile(self.__grid_ones()[:, :, np.newaxis], (1, 1, self.nAgents + self.nBaseStations)) # RESET ATTRACT PHEROMONE map

                        #Add Attract drawPheromonemap to maintain PIPE width along active routes 
                        for routeNo in parameters.Route_Details:
                            path = parameters.Route_Details[routeNo]['Route']   # route
                            path_valid = parameters.Route_Details[routeNo]['ValidRoute'] # route_validity
                            pathnodes_density = parameters.Route_Details[routeNo]['RouteNodeDensity']
                        # for id, path in enumerate(prev_paths):
                            # pathnodes_density = routes_node_density[id] 

                            if path_valid: # do topology control (TC) only if route not broken ; PIPE exists
                                
                                #Set mask box dimensions/cells - [2Tx X Tx]
                                Tx_cells = int( parameters.MAX_Transmission_Range / self.map_resolution) # SD: fixed as TX range = 10/12 cells
                                Tx_half_cells = int( Tx_cells/2 )
                                # print("At ", t, " BOX - ", Tx_cells,Tx_half_cells)

                                for ni in range(0, len(path)-1): #no attract pheromone at BS    
                                    node= path[ni]
                                    node_1hop_denisty = pathnodes_density[ni]  # excluding up and downstrean nodes

                                    # Pipe_neighbor_density_threshold = 2 #set in main_connect.py
                                    #cell where attract pheromone is dropped
                                    attract_pher_cell= tuple(np.ceil(self.Arobot[:, node] / self.map_resolution).astype(int))
                                    ii = attract_pher_cell[0]
                                    jj = attract_pher_cell[1]

                                    if 0 <= node_1hop_denisty <= self.Pipe_neighbor_density_threshold:
                                        attract_value = 0.  # changed for mulitplication

                                        # maskHorizontal=True
                                        angle_up_down_nodes= self.angle_between_points(self.Arobot[:, path[ni-1]], self.Arobot[:, path[ni+1]])
                                        if 0 <= angle_up_down_nodes%180 <45 or 135 < angle_up_down_nodes%180 <=180:
                                            maskHorizontal=False
                                        else:
                                            maskHorizontal=True
                                            
                                        #Apply Attract Mask in box dimensions/cells [2Tx X Tx]
                                        for aa in range(self.nAgents + self.nBaseStations): 
                                            if aa not in path[0:-1]:   # update BS node                                  
                                                if maskHorizontal:
                                                    self.node_pheromone_Attract[ii-Tx_cells:ii+Tx_cells, jj-Tx_half_cells:jj+Tx_half_cells , aa] = attract_value
                                                else:
                                                    self.node_pheromone_Attract[ii-Tx_half_cells:ii+Tx_half_cells, jj-Tx_cells:jj+Tx_cells , aa] = attract_value
                                        
                        #Update the node_pheromone_map maps with combining Attract maps,  that all UAVS see to take decision # COMMNETED 9/20/23; ADDED
                        self.node_pheromone_map[:, :, :] = np.multiply( self.node_pheromone_map[:, :, :] , self.node_pheromone_Attract[:, :, :])

                    ##------------------------------------------------------------------------------------------------------
                    ##---END: USE of ATTRACT PHEROMONE - (FOR TOPOLOGY CONTROL - TC_PIPE) ---
                    ##------------------------------------------------------------------------------------------------------

                # Sanity Check
                if np.any(self.node_pheromone_Repel > 1):
                    raise NameError('pheromone value in cell > 1')
                
                # Update messages
                # self.simChannel(self.channel);
                if (t % self.hello_period == 0):
                    # Update messages every hello period
                    self.simChannel(self.channel)


                # calculate connectivity
                connMat = self.connectivity(
                    self.Arobot[0:2, :self.nAgents], self.transmission_range)
                connMatwBS = self.connectivity(
                    self.Arobot[0:2, :], self.transmission_range, connect_base_stations=False)


                # Ploting UAV figures---------------------------------
                self.plot_UAV_figures(t, tidx, ax, axx, drawplots, plot_interval, drawPheromonemap,
                                    drawUAVflight, drawAirspace, drawUAVconnectivity, connMatwBS)  # BSedit


                # log stats and metrics---------------------------------
                # self.stats = self.evalutaion_metric_calc( t , connMat, self.stats)
                temp_100s, connMat100, connMat400 = self.evalutaion_metric_calc(t, connMat, connMatwBS, self.stats, temp_100s, connMat100, connMat400)

                #log UAVPositions
                self.stats.UAV_positions[t, :, :] = self.Arobot[0:2, :].T

                parameters.Node_Locations_from_MobilityModel = self.Arobot[0:2, :].T  #SHREY* Update node locations every time step (parameters.LOCATION_UPDATE_INTERVAL/1e9 sec)
                parameters.Node_NextWaypoints_from_MobilityModel = self.Acontroller[11:13, :].T   #SHREY* nxtwaypoint = (controller[11], controller[12]); Update node next waypoint locations

            elif Path(parameters.STATIC_TOPOLOGY_PATH).is_dir():
                for i in range(len(nodes)):
                    x, y = rf.nodeLocation(nodes[i].name, int(t))
                    nodes[i].latitude = x
                    nodes[i].longitude = y
            
            if t == sim_time - 1:
                self.stats.runtime.append(time.time() - starttime)
            
            # Perform updateNetworkTopology and Wait until next update
            if t < (parameters.PKT_GENERATION_START_SEC):
                yield self.env.timeout( parameters.LOCATION_UPDATE_INTERVAL ) #SHREY* Wait until next update

            elif parameters.FORCED_ROUTES or len(source_node)==len(self.multipletarget.TargetUAV_assigned):# SD*: Intinialize Route detais after t>= 1000s and after all targets found.
                if not parameters.FORCED_ROUTES:
                    parameters.Routes=[([self.multipletarget.TargetUAV_assigned[i], parameters.Gateway_Node_ID], self.multipletarget.location_xy[i].tolist()) for i in range(self.multipletarget.number_of_targets)]  # add src_dst pairs
                
                if not parameters.Route_Details:
                    print(f"time {t} ( all targets found and t>1000s) and set up src_dst pairs = {parameters.Routes}")
                    parameters.initializeROUTE_DETAILS(parameters.Routes)  
                    
                    #SD: Moved from main.py to start the processes here
                    # Sh: Queue management 
                    # MC: Setup periodic packet generation for this source
                    # Create stat log file for this flow
                    parameters.initialize_NODES_FLOWLOGS_QUEUE_PACKETGENERATION(self.env, nodes)

                yield self.env.process( rf.updateNetworkTopology(self.env, nodes) ) #SHREY* Perform updateNetworkTopology processes and Wait until next update; (do after PKT_GENERATION_START_TIME)
            
            else:
                # self.stats.skip=True   # SD: If set dont consider run for results averaging
                print("EXIT simulation; All targets not found within 1000s")
                exit(1)


            # # Perform updateNetworkTopology and Wait until next update
            # if t < (parameters.PKT_GENERATION_START_SEC):
            #     yield self.env.timeout( parameters.LOCATION_UPDATE_INTERVAL ) #SHREY* Wait until next update
            # else:
            #     yield self.env.process( rf.updateNetworkTopology(self.env, nodes) ) #SHREY* Perform updateNetworkTopology processes and Wait until next update; (do after PKT_GENERATION_START_TIME)

        return self.stats

    # gives angle between points
    @profile
    def angle_between_points(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        dx = x2 - x1
        dy = y2 - y1

        angle = math.atan2(dy, dx) * (180.0 / math.pi)

        return angle




    # function logs performance stats---------------------------------
    @profile
    def evalutaion_metric_calc(self, t: int, connMat: NDArray[np.int8], connMatwBS: NDArray[np.int8], stats: UAVStats, temp_100s: NDArray[np.floating], connMat100: NDArray[np.bool], connMat400: NDArray[np.bool]):
        # Save value every 100sec
        if (t % self.stats_interval == 0):
            # coverage percentage
            stats.percentage_coverage_in_episode = (
                int(stats.visited[1:-1, 1:-1].sum())/(stats.visited[1:-1, 1:-1].size))*100
            stats.coverage.append(stats.percentage_coverage_in_episode)
            # fairness
            stats.fairness.append(np.std(stats.frequency[1:-1, 1:-1]))
            if t >= parameters.PKT_GENERATION_START_SEC:
                stats.coverage_after_genstart.append((stats.visited_after_genstart[1:-1, 1:-1].sum()/(stats.visited_after_genstart[1:-1, 1:-1].size))*100)
                stats.fairness_after_genstart.append(np.std(stats.frequency_after_genstart[1:-1, 1:-1]))

        # # visitation frequency histogram
        # if t>0 and t % 500 == 0:
        #     vfhist_0 , bins_0=np.histogram(stats.frequency[1:-1, 1:-1], bins = np.arange(21))
        #     stats.vfhist_0.append([vfhist_0])
        # if t>1000 and t % 500 == 0:
        #     vfhist_1k , bins_1k=np.histogram(stats.frequency_after_genstart[1:-1, 1:-1], bins = np.arange(21))           
        #     stats.vfhist_1k.append([vfhist_1k])

        if t % 100 == 0:
            # recent coverage every 100sec
            stats.cellscoverage_per_100s.append(temp_100s[1:-1, 1:-1].sum())
            # self.__grid(); #SD
            temp_100s = np.zeros((len(self.xgv)+2, len(self.ygv)+2))

        # Save Connectivity output values every second
        nbins, bins, binsizes, is_biconnected_Gaint = self.myconncomp(connMat)
        # TODO: pre-allocate
        stats.no_connected_comp.append(nbins)  # NCC
        stats.largest_subgraph.append(max(binsizes)) # Largest sub component / Giant Component

        subgraphhist, _ = np.histogram(binsizes, np.arange(self.nAgents)+.5)    # get histogram of subgraph sizes
        # and add to get frequency statistics
        stats.freq_subgraph_sizes += subgraphhist

        # get avg degree of connectivity of network
        deg = connMat.sum(1)
        stats.avg_deg_conn.append(np.mean(deg, dtype=np.float64))  # AND/ANC # TODO: pre-allocate

        stats.is_biconnected_Gaint_Comp.append(is_biconnected_Gaint)

        closure = np.array(warshall(connMatwBS)) # get transitive closure of network
        stats.total_time_connected_to_BS[:, :, t] = closure[:-self.nBaseStations, self.bs_iDs] == 1
        
        if t >= parameters.PKT_GENERATION_START_SEC:
            stats.no_connected_comp_after_genstart.append(nbins)
            stats.largest_subgraph_after_genstart.append(max(binsizes))
            stats.freq_subgraph_sizes_after_genstart += subgraphhist
            stats.avg_deg_conn_after_genstart.append(np.mean(deg))
            stats.is_biconnected_Gaint_Comp_after_genstart.append(is_biconnected_Gaint)
            stats.total_time_connected_to_BS_after_genstart[:, :, t] = closure[:-self.nBaseStations, self.bs_iDs] == 1

    

        if parameters.PRINT_LOG_TOPOLOGY_METRICS:
            print(f"mob_pher.py: Time:{t} Toplogy details:"+"{"+f"NCC:{nbins}, AND:{np.mean(deg):.2f}, G:{max(binsizes)}, Nbs:{np.sum(stats.total_time_connected_to_BS[:, :, t])}"+"}")

        return temp_100s, connMat100, connMat400

    # Adjacency matrix of communcation connectivity
    @profile
    def connectivity(self, locations: NDArray[np.floating], t_range: float, connect_base_stations: bool = False) -> NDArray[np.int8]:
        D, nAgents = locations.shape
        conn: NDArray[np.int8] = ((locations.reshape((D, nAgents, 1)) -
                locations.reshape((D, 1, nAgents)))**2).sum(axis=0) < (t_range**2)
        conn[range(nAgents), range(nAgents)] = 0  # remove self-edges
        if connect_base_stations:
            for bs1, bs2 in itertools.combinations(self.bs_iDs, 2):
                conn[bs1, bs2] = 1
                conn[bs2, bs1] = 1
        return conn

    @profile
    def estimate_node_connectivity_at_nextwaypoints(self, node_id, nxt_waypoint, msgs_neighbor_nxt_waypoints, dt, t_range):
        """ Gives the node's distance weighted connectivity wrt to the node's next waypoint and neighbor nodes next waypoint.

        Args:
            node_id: node id
            nxt_waypoint:  node's next waypoint
            msgs_neighbor_nxt_waypoints: neighbour nodes next waypoints sent through hello messages
            dt: time step (NOT USED)
            t_range: transmission range ;default value-1000m

        Returns:
            dst-weighted-connectivity value :(float).

        """
        # neighbors of "node_id" using the positions in msgs
        nbrs = np.sqrt(((msgs_neighbor_nxt_waypoints[8:10, :] - arr(nxt_waypoint[0:2]).reshape(2, 1))**2).sum(0)).astype(float)
        # print(nbrs)
        d1 = nbrs <= (0.6 * t_range)
        d2 = ((nbrs > (0.6 * t_range)) & (nbrs <= t_range))
        d3 = nbrs > t_range
        nbrs[d1] = 1.0
        # nbrs[d2] * (-1.0/400.0) + 2.5 #For TX_RANGE=1000m
        nbrs[d2] = 2.5*(1. - nbrs[d2]/t_range)
        nbrs[d3] = 0.0
        # print(nbrs)
        try:
            nbrs[int(node_id)] = 0.0  # not our own neighbor
        except:
            if len(nbrs) == 0:
                return 3.0
        return nbrs.sum()

    @profile
    def node_distance_weighted_connectivity_at_current_position(self, node_id, curr_position, t_range):
        """ Gives the node's distance weighted connectivity wrt to the node's curent position and neighbor nodes current taken

        Args:
            node_id: node id
            curr_position:  node's current position coordinates[x,y]
            t_range: transmission range ;default value-1000m.

        Returns:
            dst-weighted-connectivity value :(float).

        """
        neighbor_nodes_current_positions = self.Arobot[0:2,
                                                       :]  # list of neighbour nodes current positions
        nbrs = np.sqrt(((neighbor_nodes_current_positions -
                       arr(curr_position[0:2]).reshape(2, 1))**2).sum(0)).astype(float)
        # print(nbrs)
        d1 = nbrs <= (0.6 * t_range)
        d2 = ((nbrs > (0.6 * t_range)) & (nbrs <= t_range))
        d3 = nbrs > t_range
        nbrs[d1] = 1.0
        # nbrs[d2] * (-1.0/400.0) + 2.5 #For TX_RANGE=1000m
        nbrs[d2] = 2.5*(1. - nbrs[d2]/t_range)
        nbrs[d3] = 0.0
        # print(nbrs)
        nbrs[int(node_id)] = 0.0  # not our own neighbor
        return nbrs.sum()

    @profile
    def nodes_distance_from_me_Arobot_locations(self, node_id, curr_position):
        """ Gives all the other node's distance wrt to the node's curent position and hello msgs

        Args:
            node_id: node id
            curr_position:  node's current position coordinates[x,y]

        Returns:
             other node's distance wrt to the node's curent position :(list of float values).

        """
        #      neighbor_nodes_current_positions = hello_msgs[0:2,:] # list of neighbour nodes cuurrent positions
                # list of all nodes current positionss
        all_nodes_current_positions = self.Arobot[0:2, :]
        nbrs = np.sqrt(((all_nodes_current_positions -
                       arr(curr_position[0:2]).reshape(2, 1))**2).sum(0)).astype(float)

        #      nbrs[int(node_id)] = 0.0;  # not our own neighbor
        return nbrs

    @profile
    def myconncomp(self, adj: NDArray[np.int8]):
        G: nx.Graph = cast(nx.Graph, nx.from_numpy_array(adj)) # type: ignore[reportUnknownMemberType]
        # convert generator to list if required
        CCs = list(nx.connected_components(G)) # type: ignore[reportUnknownMemberType]
        nbins = len(CCs)
        bins = np.zeros(adj.shape[0])
        binsizes = np.zeros(nbins)
        for i, c in enumerate(CCs):
            bins[list(c)] = i
            binsizes[i] = len(c)

        largest_cc = max(CCs, key=len)
        is_biconnected_Gaint_Comp = cast(bool, nx.is_biconnected(G.subgraph(largest_cc))) # type: ignore[reportUnknownMemberType]
        return nbins, bins, binsizes, is_biconnected_Gaint_Comp

    # Communications
    @profile
    def initChannel(self, nAgents):
        return np.zeros((10, nAgents, 2)) + np.nan

    @profile
    def simReceive(self, channel):
        keep = ~np.isnan(channel[0, :, 0])
        return channel[:, keep, 0]

    @profile
    def simTransmit(self, channel, uavID, txMsgs):
        channel[:, uavID, 1] = txMsgs

    @profile
    def simChannel(self, channel):
        channel[:, :, 0] = channel[:, :, 1]
        channel[:, :, 1] = np.nan

    @profile
    def gps(self, robot, noise=0):
        """A GPS is accurate to between +/- 3mon a good day"""
        return robot + (np.random.random(robot.shape)-.5)*noise

    @profile
    def decide_3d(self, uavID, controller, msgs, dt, prev_state, prev_action, t_time):
        """  """
        b = self.__boundv(dt)

        # % Agent finite state machine
        if (controller[8] == 1):
            # case 1
            # % Pick a random target way point
            curr_cell = np.ceil(controller[0:2]/self.map_resolution).astype(int)
            if self.use_pheromone:  # ATI weird, no difference
                w = self.get_neighbor_pheromone_weight(controller[7], curr_cell)
            else:
                w = self.get_neighbor_pheromone_weight(controller[7], curr_cell)

            R = np.random.choice(np.flatnonzero(w == min(w)))
            controller[11], controller[12] = self.return_next_dst_point_based_on_direction(curr_cell, R, self.hop_dist)
            controller[13] = 0
            controller[8] = 2



        # % Hover above the traget object; target way point = same as target.location_xy[:]
        elif (controller[8] == 3):
            for tt in range(self.multipletarget.number_of_targets):
                if self.multipletarget.TargetUAV_assigned[tt] == uavID:
                    controller[11], controller[12] = self.multipletarget.location_xy[tt][:] #TARGET 1
                    controller[8] = 3
                    # print("Target found by ", uavID, " at ", controller[11], controller[12])
                    return None, None



        elif (controller[8] == 2):  # % Fly to target way point
            # case 2
            # % Travel in a straight line at top speed
            controller[6] = 0
            controller[5] = b[1]  # %.maxv(dt)

            # % Turns required?
            if controller[13] > 0:
                controller[13] -= 1
            else:
                # %SHREY: turn on & face towards the target(x,y)
                [controller, steps] = self.face(controller, controller[11], controller[12], dt)
                controller[13] = steps - 1

            # % Too close to the boundary?
            # if (np.any(controller[0:2]>self.map_size) or np.any(controller[0:2] < 0)):
                # print('[Agent {}] I am OUTSIDE the boundary!'.format(controller[7]));

            # % Where will we be next timestep (after dt =1sec)?
            x, y, theta = rk4(controller[0], controller[1], controller[3], controller[5], controller[6], dt)
            nextpos = arr([x, y, theta])


            # Check if target is acquired and update target UAV information
            for tt, (target_x, target_y) in enumerate(self.multipletarget.location_xy):

                # Compute Euclidean distance between UAV and target
                distance = np.hypot(nextpos[0] - target_x, nextpos[1] - target_y)  # np.hypot(x1,x2) = sqrt(x1**2 + x2**2)

                # Check if target is within range and not yet serviced
                if distance < self.waypoint_radius + 50 and not self.multipletarget.service_status[tt]:
                    
                    # Assign UAV to the target
                    self.multipletarget.TargetUAV_assigned[tt] = uavID
                    self.multipletarget.TargetUAV_assigned_Time[tt] = t_time
                    self.multipletarget.service_status[tt] = True

                    # Update controller information
                    controller[11:13] = target_x, target_y
                    controller[8] = 3  # flag set to hover above the traget
                    
                    print(f"Target-{tt} found by UAV-{uavID} at ({target_x}, {target_y}), at time {t_time}")
                    
                    return None, None  # Exit early once a target is assigned                

            # % SHREY : random direction selection untill it turns away from boundary
            # SHREY: FIX if (np.any(nextpos[0:2] > self.map_size-self.turn_buffer) or np.any(nextpos[0:2]<0+self.turn_buffer)):
            if (np.any(nextpos[0:2] > self.map_size-self.turn_buffer) or np.any(nextpos[0:2] < 0+self.turn_buffer)):
                controller[8] = 1

            if self.collision_avoidance:
                # % Check too close to another agent?
                [controller, stop] = self.airspace(
                    controller, nextpos, msgs, dt)
                if stop:
                    return None, None

            # % When we have arrived just before the next way point ***, then do the following:
            # % (Note:Use a threshold to stop the UAV spinning around the targ.)

            if (np.sqrt(((nextpos[0] - controller[11])**2) + ((nextpos[1] - controller[12])**2)) < self.waypoint_radius):
                # curr_cell = tuple(np.ceil(controller[0:2]/self.map_resolution).astype(int) - 1); # SHREY: BUG-FIX subtract by 1 due to different indexing
                curr_cell = np.ceil(
                    controller[0:2]/self.map_resolution).astype(int)
                # if(np.any(arr(curr_cell)<1) or np.any(arr(curr_cell)>60)): print(curr_cell, controller[0:2])
                # % Get the pheromone weight in each direction
                if self.use_pheromone:
                    wgt_avg_pheromone_values = self.get_neighbor_pheromone_weight(
                        controller[7], curr_cell)
                else:
                    wgt_avg_pheromone_values = np.zeros(
                        (1, 8))  # Random direction selection

                # % Choose the direction based on pheromone or pheromone+connectivity condition.
                if self.use_connect:
                    curr_heading_R = int((controller[3] + 22.5)/45) % 8

                    if (self.fwd_scheme == 3):   # find pheromone directions
                        waypoint_directions = arr([curr_heading_R-1, curr_heading_R, curr_heading_R+1]) % 8
                        wgt_avg_pheromone_values_at_nxt_waypoints = wgt_avg_pheromone_values[waypoint_directions]

                    elif (self.fwd_scheme == 5):  # or:
                        waypoint_directions = arr([curr_heading_R-2, curr_heading_R-1, curr_heading_R, curr_heading_R+1, curr_heading_R+2]) % 8
                        wgt_avg_pheromone_values_at_nxt_waypoints = wgt_avg_pheromone_values[waypoint_directions]

                    else:
                        raise ValueError('code suports only fwd=3 or 5')

                    connectivity_at_nxt_waypoints = []  # list indicating node connectivity at nextwaypoint locations
                    bs_connected_at_nxt_waypoints = []  # BSedit: list indicating BS connectivity at nextwaypoint locations
                    dist_to_bs = []  # BSedit

                    uav_pos_now = np.copy(self.Arobot[0:2, :])
                    connMat_now = self.connectivity(
                        uav_pos_now, self.transmission_range)

                    for direction_R in waypoint_directions:
                        nxt_waypoint = [None, None]
                        nxt_waypoint[0], nxt_waypoint[1] = self.return_next_dst_point_based_on_direction(
                            curr_cell, direction_R, self.hop_dist)             # get possible nextwaypoint locations (X,Y) based on current heading direction
                        dist_wt_connectivity = self.estimate_node_connectivity_at_nextwaypoints(
                            controller[7], nxt_waypoint, msgs, dt, self.transmission_range)         # get connectivity at nextwaypoint flag
                        connectivity_at_nxt_waypoints.append(
                            dist_wt_connectivity)
                        connections, _ = self.base_station_connections_from( uavID, msgs, nxt_waypoint )
                        bs_connected_at_nxt_waypoints.append(connections)         # BSedit: How many BS connections are at nextwaypoint flag

                        dbs = []
                        for i in range(self.nBaseStations):
                            dbs.append( np.sqrt( np.sum( np.square(arr(self.bs_positions[i])-arr(nxt_waypoint[0:2])) )).astype(float) )
                        dist_to_bs.append(min(dbs))
                    
                    best_connection = max(bs_connected_at_nxt_waypoints)
                    if best_connection == 0:
                        bs_connected_at_nxt_waypoints = [False] * len(bs_connected_at_nxt_waypoints)
                    else:
                        best_connection = min(best_connection, self.decision_bs_connections)
                        bs_connected_at_nxt_waypoints = [connection >= best_connection for connection in bs_connected_at_nxt_waypoints]

                    bs_connected_at_nxt_waypoints = np.array(bs_connected_at_nxt_waypoints)  # BSedit

                    # getting the current state_________________________________________________________________________
                    state_weighted_avg_pheromone = wgt_avg_pheromone_values_at_nxt_waypoints
                    # Weighted degree of connectivity at next waypoints at t+dt
                    state_connectivity = connectivity_at_nxt_waypoints

                    state_conn = np.copy(arr(state_connectivity).astype(int))
                    state_conn[np.where(state_conn > 10)] = 10
                    for i in state_conn:
                        parameters.MOBILITY.connectivity_histogram[i] += 1

                    # current_state = state_weighted_avg_pheromone
                    current_state = np.concatenate((arr(state_weighted_avg_pheromone), arr(state_connectivity)))  # state-s

                    # ATI: improve?
                    Rdeg = np.copy(connectivity_at_nxt_waypoints)   # connectivity at nxtwayoints
                    pher_ww = np.copy(wgt_avg_pheromone_values_at_nxt_waypoints)   # pheroomone at nxtwayoints

                    alpha = []
                    alpha = list(map(self.alpha_func, Rdeg))
                    alpha = np.array(alpha)

                    pher_ww = np.clip(pher_ww, 0, 1)
                    if np.any(pher_ww > 1):
                        # pher_ww=np.clip(pher_ww, 0, 1)#raise ValueError("pher_ww > 1")
                        raise ValueError("pher_ww > 1")

                    Pi = (alpha*(1-pher_ww))
                    # Pi=Pi/np.sum(Pi)

                    # CHECK BS connectivity True or False  #BSedit
                    # will not be connect at any of the next possible waypoints
                    if np.all(bs_connected_at_nxt_waypoints == False):

                        # if no route at next waypoint select waypoint towards current routes to bs nexthop node's next waypoint.
                        hop_lengths = self.current_hoplengths_to_BS( connMat_now, uavID )
                        BS_connected_flag_now = False
                        cur_min = np.inf
                        for idx, hop in hop_lengths:
                            if hop != np.inf:
                                BS_connected_flag_now = True
                                if hop < cur_min:
                                    cur_min = hop
                                    bs_idx = idx
                        
                        if BS_connected_flag_now:
                            nxt_hop = shortest_path(connMat_now, source=uavID, target=self.bs_iDs[bs_idx])[1]
                            # print(nxt_hop, shortest_path(G_aftertisec, source=uavID, target=self.bs_iD))
                            # print("nobreak---", np.linalg.norm(uav_pos_now[0:2,uavID]-self.bs_position))

                            if nxt_hop == self.bs_iDs[bs_idx]:  # next hop is BS itself

                                # Move next waypoint closest to BS
                                bs_list = np.zeros(5)
                                bs_list[np.argmin(dist_to_bs)] = 1
                                Pi = bs_list

                            else:  # next hop is a  neighbor with connection to BS
                                # move to  this neighbor next waypoint
                                dist_to_nexthopnode = []
                                for direction_R in waypoint_directions:
                                    nxt_waypoint = [None, None]
                                    nxt_waypoint[0], nxt_waypoint[1] = self.return_next_dst_point_based_on_direction(
                                        curr_cell, direction_R, self.hop_dist)

                                    # d_nxthop=np.sqrt( np.sum( np.square(arr(uav_pos_now[0:2,nxt_hop])-arr(nxt_waypoint[0:2])) )).astype(float)  # next hop node current position
                                    nxthopnode_nextwaypoint = msgs[8:10, nxt_hop]
                                    # # next hop node next waypoint position
                                    d_nxthop = np.sqrt(np.sum(
                                        np.square(arr(nxthopnode_nextwaypoint)-arr(nxt_waypoint[0:2])))).astype(float)
                                    dist_to_nexthopnode.append(d_nxthop)

                                nxthop_list = np.zeros(5)
                                nxthop_list[np.argmin(dist_to_nexthopnode)] = 1
                                Pi = nxthop_list

                        #   bs_list = np.zeros(5)
                        #   bs_list[np.argmin(dist_to_bs)]=1
                        #   Pi=bs_list # force Pi of i with min dist_to_BS to 1 and rest to 0.

                    else:
                        #   dist_to_bs = 0.5 + ((dist_to_bs - np.min(dist_to_bs)) * 0.5) / (np.max(dist_to_bs) - np.min(dist_to_bs)) # scale 0.5 to 1
                        #   bs_list = dist_to_bs * bs_connected_at_nxt_waypoints
                        #   Pi=Pi*bs_list

                        Pi = Pi*bs_connected_at_nxt_waypoints

                    #   if np.all(Pi==0):
                    #       if np.any(Rdeg >= self.min_degree):
                    #           current_action = np.where(Rdeg >= self.min_degree)
                    #       else:
                    #           current_action = np.where(Rdeg == max(Rdeg))
                    #   else:
                    #     #Pi=Pi/np.sum(Pi)
                    #     current_action = np.where(Pi == max(Pi))

                    current_action = np.where(Pi == max(Pi))

                    current_action = current_action[0]
                    current_action = np.random.choice(current_action)
                    direction_R = waypoint_directions[current_action]

                    ##########################################################################################################################
                    # get way ponit of next dst hop center
                    controller[11], controller[12] = self.return_next_dst_point_based_on_direction(
                        curr_cell, direction_R, self.hop_dist)

                    return current_state, current_action

                else:  # % pheromone/rand model (NOT USED DURING TRAINING)
                    raise NameError(
                        'Not supposed to be here during training run')

        else:  # % Something went wrong
            # otherwise
            raise NameError(
                '[Agent %d] I am in an unknown state. Help!', controller[7])

        return None, None

    @profile
    def current_hoplengths_to_BS(self, connMat, uavID):  #BSedit
        hop_lengths = []
        distances = shortest_path_length(connMat, source=uavID)
        
        for i in range(len(self.bs_iDs)):
            hop_lengths.append((i, distances[self.bs_iDs[i]]))
        return hop_lengths

    @profile
    def base_station_connections_from(self, uavID, msgs, nxt_waypoint):  # BSedit
        # BS scheme FUNCTION
        uav_pos_next = np.copy(msgs[8:10, :])
        uav_pos_next[0:2, uavID] = nxt_waypoint
        # print("uav_pos_next", uav_pos_next[0:2, uavID])
        connMat_next = self.connectivity(uav_pos_next, self.transmission_range)
        return self.connected_to_BS(connMat_next, uavID)

    @profile
    def connected_to_BS(self, connMat, uavID):
        hop_lengths = self.current_hoplengths_to_BS(connMat, uavID)
        connections = 0
        for _, hop in hop_lengths:
            if hop != np.inf:
                connections += 1
        return connections, hop_lengths

    @profile
    def alpha_func(self, deg): # Tuning function 
        if deg <= self.alpha_type:
            alp = deg / self.alpha_type
        elif self.alpha_type < deg <= 3:
            alp = 1.
        else:
            alp = 1./3
        return alp


    # % Are we too close to another agent?
    @profile
    def airspace(self, controller, nextpos, msgs, dt):
        stop = False

        them = 0
        index = 0
        closest = np.inf

        # ATI: do a simple thresholding before running through the messages, based on current position etc.
        # Only check messages (& estimate projected position) from UAVs that are already "close enough"
        init_dist2 = ((nextpos[0:2, np.newaxis]-msgs[0:2, :])**2).sum(0)
        check = np.where(
            init_dist2 < 16*(self.uav_airspace+self.__boundv(dt)[1])**2)[0]

        for jj in check:  # range(len(msgs)):
            # % Get the other agent, skipping ourself
            if jj == controller[7]:
                continue
            other = msgs[:, jj]  # %msgs{jj}

            # % If it continued, where would it be?
            # % SHREY: estimate the position of the neighbor agents from their previous 5 positions
            x, y, theta = rk4(other[0], other[1],
                              other[3], other[5], other[6], dt)

            # % Too close to our projected location?
            # % SHREY: nextpos(x,y) is my future position estimate
            dist = ((nextpos[0:2]-[x, y])**2).sum().round()
            # if dist < closest and dist < 2*self.uav_airspace:  # ATI: added 2nd condition vvv
            if dist < closest and dist < 2*(self.uav_airspace**2):
                them = arr([x, y, theta])
                dx = nextpos[0]-them[0]
                dy = nextpos[1]-them[1]
                next_v = arr([dx, dy])
                # next_v = (next_v/np.linalg.norm(next_v) )* 60;
                next_v = (next_v / np.sqrt(np.sum(next_v**2))) * \
                    60  # Note next_v vector should not be zero
                xa = nextpos[0]-controller[0]
                ya = nextpos[1]-controller[1]

                push_v = arr([xa, ya])
                # push_v = (push_v/np.linalg.norm(push_v) )* 60;
                push_v = (push_v / np.sqrt(np.sum(push_v**2))) * \
                    60  # Note push_v vector should not be zero

                # combined_v = next_v+push_v
                combined_v = np.add(next_v, push_v)
                controller[11] = controller[0]+combined_v[0]
                controller[12] = controller[1]+combined_v[1]
                # plot(controller[11],controller[12],'bx');

                controller[13] = 0
                stop = True

        return controller, stop

    # % Too close?

    @profile
    def proximity(self, controller, nextpos, them, index, closest, radius):
        stop = False

        # % Is there a closest neighbour?
        if closest < np.inf:
            # % If same neighbour as before & further away, leave as-is
            if index == controller[9]:
                if closest > controller[10]:
                    return controller, stop
            dx = nextpos[0]-them[0]
            dy = nextpos[1]-them[1]
            next_v = arr([dx, dy])
            # next_v = (next_v/np.linalg.norm(next_v) )* 60;
            next_v = (next_v / np.sqrt(np.sum(next_v**2))) * \
                60  # Note next_v vector should not be zero

            xa = nextpos[0]-controller[0]
            ya = nextpos[1]-controller[1]

            push_v = arr([xa, ya])
            # push_v = (push_v/np.linalg.norm(push_v) )* 60;
            push_v = (push_v / np.sqrt(np.sum(push_v**2))) * \
                60  # Note push_v vector should not be zero

            combined_v = np.add(next_v, push_v)  # combined_v = next_v+push_v
            controller[11] = controller[0]+combined_v[0]
            controller[12] = controller[1]+combined_v[1]
            # plot(controller[11],controller[12],'bx');

            controller[13] = 0
            stop = True

        return controller, stop

    # % Turn towards coordinates
    @profile
    def face(self, controller, x, y, dt):
        # SHREY: atan2d returns angle between [-180, 180] degerees
        delta = (180/np.pi)*np.arctan2(x -
                                       controller[0], y-controller[1]) - controller[3]
        controller, steps = self.turn(controller, delta, dt)
        return controller, steps

    # % Picks a random target
    @profile
    def randomtarg(self, controller, lowerX, upperX, lowerY, upperY):
        # % Pick a random location, between bounds
        a = max(lowerX, 0)
        b = min(upperX, self.map_size)
        r = (b-a)*np.random.rand() + a
        controller[11] = round(r)
        a = max(lowerY, 0)
        b = min(upperY, self.map_size)
        r = (b-a)*np.random.rand() + a
        controller[12] = round(r)

        # %     angle=controller(4); %rand*360;
        # %     xdot = 40*sind(angle)
        # %     ydot = 40*cosd(angle)
        # %     controller(12)=(controller(1) + xdot)
        # %     controller(13)=(controller(2) + ydot)
        return controller

    # % Turn by delta degrees
    @profile
    def turn(self, controller, delta, dt):
        b = self.__boundv(dt)                  # Make bounds object
        delta = (delta+180) % 360 - 180          # Normalise the angle

        if delta == 0:   # Nothing to do?
            steps = 1
            return controller, steps

        # % Delta is now in deg/s
        # % SHREY : multiplying by 2, 3.. for mor steps and reduce max turn in bounds.m
        delta = delta / (1*dt)

        # % Something to do!
        # ATI closed form calculation
        steps = np.ceil(abs(delta) / b[3] / b[1])
        v = max(np.ceil(abs(delta) / steps / b[3]), b[0])
        # find integer s,v : b(1) < v < b(2), b(3) < mu < b(4) & delta = mu*v*s
        mu = delta / steps / v

        controller[5] = v
        controller[6] = mu
        return controller, steps

    @profile
    def move(self, robot, v, mu, dt):
        # Make bounds object
        b = self.__boundv(dt)

        v = min(max(v, b[0]), b[1])    # Physically cap speed
        mu = min(max(mu, b[2]), b[3])  # Physically cap turn

        # Runge-Kutta (rk4)
        robot[0], robot[1], robot[3] = rk4(
            robot[0], robot[1], robot[3], v, mu, dt)
        return robot

    @profile
    def mod_circcirc_octave(self, x1, y1, r1, x2, y2, r2):
        P1 = arr([x1, y1])
        P2 = arr([x2, y2])
        d2 = sum((P2-P1)**2)

        P0 = (P1+P2)/2+(r1**2-r2**2)/d2/2*(P2-P1)
        t = ((r1+r2)**2-d2)*(d2-(r2-r1)**2)
        if t <= 0:
            # print("The circles don't intersect.\n")
            xout = arr([np.nan, np.nan])
            yout = arr([np.nan, np.nan])
        else:
            T = np.sqrt(t)/d2/2*arr([[0, -1], [1, 0]]).dot(P2-P1)
            Pa = P0 + T  # % Pa and Pb are circles' intersection points
            Pb = P0 - T

            xout = arr([Pa[0], Pb[0]])
            yout = arr([Pa[1], Pb[1]])
        return (xout, yout)

    """
  function [xout,yout]=circcirc_FAST(x1,y1,r1,x2,y2,r2)
      %CIRCCIRC  Intersections of circles in Cartesian plane
      %
      %  [xout,yout] = CIRCCIRC(x1,y1,r1,x2,y2,r2) finds the points
      %  of intersection (if any), given two circles, each defined by center
      %  and radius in x-y coordinates.  In general, two points are
      %  returned.  When the circles do not intersect or are identical,
      %  NaNs are returned.  When the two circles are tangent, two identical
      %  points are returned.  All inputs must be scalars.
      %
      %  See also LINECIRC.

      % Copyright 1996-2007 The MathWorks, Inc.
      % $Revision: 1.10.4.4 $    $Date: 2007/11/26 20:35:08 $
      % Written by:  E. Brown, E. Byrns

      r3=sqrt((x2-x1).^2+(y2-y1).^2);

      indx1=find(r3>r1+r2);  % too far apart to intersect
      indx2=find(r2>r3+r1);  % circle one completely inside circle two
      indx3=find(r1>r3+r2);  % circle two completely inside circle one
      indx4=find((r3<10*eps)&(abs(r1-r2)<10*eps)); % circles identical
      indx=[indx1(:);indx2(:);indx3(:);indx4(:)];

      anought=atan2((y2-y1),(x2-x1));

      %Law of cosines

      aone=acos(-((r2.^2-r1.^2-r3.^2)./(2*r1.*r3)));

      alpha1=anought+aone;
      alpha2=anought-aone;

      xout=[x1 x1]+[r1 r1].*cos([alpha1 alpha2]);
      yout=[y1 y1]+[r1 r1].*sin([alpha1 alpha2]);

      % Replace complex results (no intersection or identical)
      % with NaNs.

      if ~isempty(indx)
          xout(indx,:) = NaN;
          yout(indx,:) = NaN;
      end
  end
  """

    # % gets the pheromone value of neighbouring cells
    @profile
    def get_neighbor_pheromone_weight(self, node_id, curr_cell):
        """Extract neighboring cell pheromone information"""

        if (self.hop_dist == 1):
            H = arr([[1]])            # define filter for pheromones
        else:
            # sIMPLIED FOR TRAINING A SIMPLE Q-MODEL    LATER:??? CHANGE
            H = (1./12)*arr([[1., 1., 1.], [1., 4., 1.], [1., 1., 1.]])
        #   H = arr([[1]]) ??
        # W = int(self.hop_dist * (max(H.shape)+1)/2);        # get filter maximum width
        W = int(self.hop_dist + (max(H.shape)+1)/2)
        buffer_value = 4                              # "beyond edge" values

        extracted = np.zeros((2*W+1, 2*W+1)) + \
            buffer_value  # store extracted sub-map
        N0, N1, _ = self.node_pheromone_map.shape
        i, j = int(curr_cell[0]), int(curr_cell[1])
        i0, i1 = max(i-W, 0), min(i+W+1, N0)
        ii0 = i0-i+W
        ii1 = 2*W+1 + i1-i-W-1
        j0, j1 = max(j-W, 0), min(j+W+1, N1)
        jj0 = j0-j+W
        jj1 = 2*W+1 + j1-j-W-1
        try:
            extracted[ii0:ii1, jj0:jj1] = self.node_pheromone_map[i0:i1,
                                                                  j0:j1, int(node_id)]
        except ValueError:
            print(curr_cell, "\n")
            print(ii0, ii1, jj0, jj1, " <- ", i0, i1, j0, j1)

        #   temp = convolve2d( extracted, H, boundary='symm', mode='same') # temp = convolve2d(extracted,H,'same');
        # temp = convolve2d(extracted,H,'same');
        temp = convolveim(extracted, H, mode='constant')

        weight = temp[W+self.hop_dist*arr([0, 1, 1, 1, 0, -1, -1, -1]),
                      W+self.hop_dist*arr([1, 1, 0, -1, -1, -1, 0, 1])]

        # if np.any(weight>1):
        # raise ValueError("get_neighbor_pheromone_weight return value >1")
        return weight

    @profile
    def return_next_dst_point_based_on_direction(self, curr_cell, R, hop_dist):
        controller_target_x, controller_target_y = None, None
        # ATI replacement code
        # sub = 1 if hop_dist==5 else 0;           # what is this?  % SHREY: BUG_FIX commented uwanted lines
        # if (hop_dist % 2 == 1): hop_dist -= sub; # ^^ ???
        xoff = [0, 1, 1, 1, 0, -1, -1, -1]
        yoff = [1, 1, 0, -1, -1, -1, 0, 1]
        controller_target = curr_cell + arr([xoff[R], yoff[R]])*hop_dist
        # controller_target = (controller_target + 1 - .5) * self.map_resolution # SHREY: BUG_FIX replaced controller_target = (controller_target - .5) * self.map_resolution
        controller_target = (controller_target - .5) * self.map_resolution
        # print(controller_target)
        return controller_target

    # function merge pheromone map of connected UAV neighbors
    @profile
    def merge_pheromone_map(self, uavID, nbrs):
        self.node_pheromone_Repel[:, :, nbrs] = np.maximum(self.node_pheromone_Repel[:, :, nbrs], self.node_pheromone_Repel[:, :, uavID:uavID+1])

    @profile
    def plot_UAV_figures(self, t, tidx, ax, axx, drawplots, plot_interval, drawPheromonemap, drawUAVflight, drawAirspace, drawUAVconnectivity, connMat):  # BSedit
        # if (drawplots and ((t % plot_interval) == 0)):
        if drawplots and (t >= parameters.PKT_GENERATION_START_SEC):
            figure, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes

            # Make colour
            color = self.colors[np.arange(self.nAgents + self.nBaseStations).astype(int) % self.colors.shape[0], :]

            if drawUAVflight:
                # plot robot locations
                plt.scatter(self.Arobot[0, :]/self.map_resolution, self.Arobot[1, :]/self.map_resolution, c=color)
                plt.axis([0, self.map_size/self.map_resolution, 0, self.map_size/self.map_resolution])
                for uavID in range(self.nAgents + self.nBaseStations):
                    plt.text(self.Arobot[0, uavID]/self.map_resolution, self.Arobot[1,uavID]/self.map_resolution, str(uavID))

                plt.grid(True)
                # for uavID in range(self.nAgents + self.nBaseStations):
                #     xyz = np.vstack((self.Arobot_history[tidx % self.nHistory+1:, :, uavID],
                #                     self.Arobot_history[:tidx % self.nHistory, :, uavID], self.Arobot[np.newaxis, 0:3, uavID]))
                #     plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], ':', c=color[uavID, :])
                plt.plot(self.Acontroller[11, :]/self.map_resolution, self.Acontroller[12, :]/self.map_resolution, 'cx')
                for bsIdx in range(self.nBaseStations):
                    plt.plot([self.bs_positions[bsIdx][0]/self.map_resolution], [self.bs_positions[bsIdx][1]/self.map_resolution], marker='s')  # BSedit
                for i in self.multipletarget.location_xy:
                    # print(i[0] , i[1], 0)
                    plt.plot(  [i[0]/self.map_resolution] , [i[1]/self.map_resolution], 'rs')  # BSedit

            rr, cc = np.nonzero(connMat)
            for i, j in zip(rr, cc):
                plt.plot([self.Arobot[0, i]/self.map_resolution, self.Arobot[0, j]/self.map_resolution], [self.Arobot[1, i]/self.map_resolution, self.Arobot[1, j]/self.map_resolution],
                          'b-', lw=.5)  # BSedit
            if drawPheromonemap:
                plt.imshow(self.node_pheromone_map[:, :, self.bs_iDs[0]].T, vmin=0, vmax=1)
                # plt.colorbar()
            plt.title('Simulation state at t={} secs'.format(t))
            # plt.pause(.01)
            figure.savefig(f"{parameters.PHEROMONE_GRAPHS}{t}.png", bbox_inches='tight', dpi=250)
            plt.close(figure)


warshall = None
@jit(nopython=True)
def warshall_py(adj_matrix):
    n = adj_matrix.shape[0]
    closure = adj_matrix.astype(np.bool_)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
    
    return closure

shortest_path_length = None
@profile
def shortest_path_length_py(G, source):
    n = len(G)
    
    distance = np.full(n, np.inf)
    distance[source] = 0
    
    dq = deque([source])
    
    while dq:
        cur_node = dq.popleft()
        
        for nei in range(n):
            if G[cur_node, nei] and distance[nei] == np.inf:
                distance[nei] = distance[cur_node] + 1
                dq.append(nei)
    
    return distance

shortest_path = None
@profile
def shortest_path_py(connMat, source, target):
    G = nx.from_numpy_array(connMat)
    return nx.shortest_path(G, source=source, target=target)


rk4 = None
@profile
def rk4_py(x, y, theta, v, mu, dt):
    """ Runge-Kutta (rk4) """

    def f_continuous(theta, v, mu):
        xdot = v*np.sin(theta)
        ydot = v*np.cos(theta)
        thetadot = v*mu
        return xdot, ydot, thetadot

    thetar = theta/180*np.pi
    mur = mu/180*np.pi
    k1_x, k1_y, k1_theta = f_continuous(thetar, v, mur)
    k2_x, k2_y, k2_theta = f_continuous(thetar + k1_theta*dt/2, v, mur)
    k3_x, k3_y, k3_theta = f_continuous(thetar + k2_theta*dt/2, v, mur)
    k4_x, k4_y, k4_theta = f_continuous(thetar + k3_theta*dt, v, mur)

    x = x + (k1_x+2*k2_x+2*k3_x+k4_x)*dt/6
    y = y + (k1_y+2*k2_y+2*k3_y+k4_y)*dt/6
    theta = theta + 180/np.pi*(k1_theta+2*k2_theta+2*k3_theta+k4_theta)*dt/6

    theta = theta % 360
    return x, y, theta

try:
    from phero_c import rk4_cy, shortest_path_length_cy, warshall_cy, shortest_path_cy
    rk4 = rk4_cy
    shortest_path_length = shortest_path_length_cy
    warshall = warshall_cy
    shortest_path = shortest_path_cy
except Exception as e:
    print("pheromone.py: error loading cython file; defining in pure python (slower)")
    print("  to build, run:  python3 setup.py build_ext --inplace ")
    rk4 = rk4_py
    shortest_path_length = shortest_path_length_py
    warshall = warshall_py
    shortest_path = shortest_path_py