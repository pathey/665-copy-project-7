'''
# Sh: File contains funcitonalities required to update node location, generate updated network topology and find new routes.
# last modified on: Nov 11, 2024
'''
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, ParamSpec, TypeVar, TypedDict, TYPE_CHECKING

from numpy.typing import NDArray
import ether
import parameters
import os
import linecache
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import re
from PIL import Image
import math
from scipy.constants import pi
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import TransportLayer as tl # Sh: Mimic packet reinsertion functionality at the source node if dropped at the interemdiate nodes due to sudden route change.

from line_profiler import profile # type: ignore[reportAssignmentType]
import simpy

if TYPE_CHECKING:
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

ProcessedPathEntry = TypedDict("ProcessedPathEntry", {'Path': list[int], 'HC': int, 'IL': int, 'Max Data Rate Supported': float,
								'HQ Path': bool, 'Tangled Path': bool, 'Cost': float})

# Create network graph
@profile
def createTopology (node_locations: dict[int, tuple[float, float]], max_transmission_ranges: dict[int, float], node_max_TRANSMISSION_ranges: dict[int, float], curr_tx_ranges: dict[int, dict[int, float]], weighted_link_capacities: dict[int, dict[int, float]], curr_sim_time: int, display: bool = True):
    G: 'nx.Graph[int]' = nx.Graph()    # Empty graph
    G_plot: 'nx.Graph[int]' = nx.Graph()    # Empty graph for plotting (without GATEWAY node and edges)

    # Add nodes to the graph with positions
    for eachNode, location in node_locations.items():
        #nodeId = int(re.search(r'\d+', eachNode).group()) # Id without name
        G.add_node(eachNode, pos=location)
        if display: G_plot.add_node(eachNode, pos=location)
    
    if not parameters.FORCED_ROUTES and parameters.MOBILITY.ROUTING_COST_TYPE != 3:
        G.add_node(parameters.Gateway_Node_ID) # Add gateway node to the graph
        for bs_id in parameters.BSIDs:
            G.add_edge(parameters.Gateway_Node_ID, bs_id, weight=0) # Add edges between gateway and base stations

    # Add edges based on transmission ranges
    for node1, pos1 in node_locations.items():
        for node2, pos2 in node_locations.items():
            if node1 != node2:
                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

                if parameters.CHANNEL_AWARE_ROUTE_COMPUTATION:
                    # Sh: Ensure symmetric link in a noisy enviornment
                    if (distance <= min(curr_tx_ranges[node1][node2], curr_tx_ranges[node2][node1])):
                        G.add_edge(node1, node2, weight=max(weighted_link_capacities[node1][node2],weighted_link_capacities[node2][node1]))
                        if display: G_plot.add_edge(node1, node2, weight=max(weighted_link_capacities[node1][node2],weighted_link_capacities[node2][node1]))
                else:
                    # Sh: Use GPS location to decide a symmetric link
                    if distance <= min(max_transmission_ranges[node1],max_transmission_ranges[node2]):
                        G.add_edge(node1, node2, weight=max(weighted_link_capacities[node1][node2],weighted_link_capacities[node2][node1]))
                        if display: G_plot.add_edge(node1, node2, weight=max(weighted_link_capacities[node1][node2],weighted_link_capacities[node2][node1]))
    
    # Visualize based on current tx power and rate
    if display:
        plot_topology(G_plot,max_transmission_ranges,curr_sim_time, node_max_TRANSMISSION_ranges) # Note: set parameter display=True for plotting
    return G

@profile
def plot_topology(graph: 'nx.Graph[int]', max_transmission_ranges: dict[int, float], curr_sim_time: int, node_max_TRANSMISSION_ranges: dict[int, float]):
    # Get node positions from the 'pos' attribute
    pos: dict[int, tuple[float, float]] = nx.get_node_attributes(graph, 'pos') # type: ignore[reportUnknownMemberType]

    # Draw the graph
    figure, ax = plt.subplots() # type: ignore[reportUnknownMemberType]
    nx.draw(graph, pos, nodelist=[i for i in range(parameters.NUMBER_OF_NODES)], with_labels=True, node_size=100, node_color='black', font_size=8, font_color='white', font_weight='bold', edge_color='gray', linewidths=1, alpha=0.7, ax=ax)
    
    # Draw routes with different colors if route_details provided
    if parameters.Route_Details:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        for route_idx, route_info in parameters.Route_Details.items():
            if route_info.get('ValidRoute', False):
                route = route_info['Route']
                color = colors[route_idx % len(colors)]
                
                # Draw the route path
                for i in range(len(route) - 1):
                    node1 = route[i]
                    node2 = route[i + 1]
                    if node1 in pos and node2 in pos:
                        x1, y1 = pos[node1]
                        x2, y2 = pos[node2]
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.8,  # type: ignore
                               label=f'Route {chr(ord('A')+route_idx)}: {route[0]}-{route[-1]}' if i == 0 else "")

    # Note: node size should be lower than its transmission range for it to be seen in plot

    # Customize the graph
    ax.axis("on")
    ax.set_xlim(0,parameters.AREA_X)
    ax.set_ylim(0,parameters.AREA_Y)
    ax.set_xlabel("Latitude (in m)") # type: ignore
    ax.set_ylabel("Longitude (in m)") # type: ignore
    ax.set_title(f"Time: {curr_sim_time}s, #Nodes: {parameters.NUMBER_OF_NODES-1}") # type: ignore
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True) # type: ignore

    # Draw transmission range circles using node_max_TRANSMISSION_ranges (actual transmission power ranges)
    # Only draw circles for nodes participating in routes
    route_nodes: set[int] = set()
    if parameters.Route_Details:
        for route_info in parameters.Route_Details.values():
            if route_info.get('ValidRoute', False):
                route_nodes.update(route_info['Route'])
    
    # Only draw transmission range for nodes participating in routes
    Circles = [plt.Circle(location, node_max_TRANSMISSION_ranges[eachNode]) for eachNode, location in pos.items() if eachNode in route_nodes] # type: ignore
    ax.add_collection(PatchCollection(Circles, facecolor='deepskyblue', alpha=0.1))

    if route_nodes:
        plt.legend([plt.Circle((0, 0), 0, color='deepskyblue', alpha=0.1)], ["Route node transmission range"], loc='upper right') # type: ignore
    else:
        plt.legend([plt.Circle((0, 0), 0, color='deepskyblue', alpha=0.1)], ["Max transmission range"], loc='upper right') # type: ignore

    # Add legend for routes
    if parameters.Route_Details:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8) # type: ignore

    # Save it
    figure.savefig(f"{parameters.NETWORK_GRAPHS}{curr_sim_time}.png", bbox_inches='tight', dpi=250) # type: ignore

    # Display the graph
    # plt.show()
    plt.close(figure)

@profile
def createGif():
    allFiles = [eachFile for eachFile in os.listdir(parameters.NETWORK_GRAPHS) if eachFile.endswith('.png')]
    if allFiles:
        # Sort by numeric prefix
        allFiles.sort(key=lambda x: int(x.split('.')[0]))
        frames: list[Image.Image] = []
        for eachFile in allFiles:
            frames.append(Image.open(os.path.join(parameters.NETWORK_GRAPHS,eachFile)))
    
        # Convert to GIF and save
        output_path = os.path.join(parameters.NETWORK_GRAPHS, "NetworkTopology.gif")
        frames[0].save( output_path, save_all=True, append_images=frames[1:], duration=500, loop=0) # Set loop to 0 for an infinite loop
    return

# ANDRES: stores the transmission power in the correspondent distance interval set (no duplicates)
@profile
def store_transmission_power(distance: float, tx_power: float) -> None:
    interval_index = int(distance // 100)
    #If this interval has not been used yet, we create a new set 
    if interval_index not in parameters.DISTANCE_TRANSMISSION_POWER_DICT:
        parameters.DISTANCE_TRANSMISSION_POWER_DICT[interval_index] = set()
    # Add the transmission power to the corresponding set
    parameters.DISTANCE_TRANSMISSION_POWER_DICT[interval_index].add(tx_power)
    return

# ANDRES: creates an image with the plot of the graph with the current values in the dictionary
@profile
def plot_distance_txPower_graph(curr_sim_time: int):
    # We get the total number of intervals and round up to the nearest hundred 
    interval_count = math.ceil(parameters.MAX_Transmission_Range / 100.0)
    max_value_rounded = interval_count*100

    # Lists to store all the points 
    all_x: list[float] = []
    all_y: list[float] = []
    min_x: list[float] = []
    min_y: list[float] = []
    max_x: list[float] = []
    max_y: list[float] = []
    avg_x: list[float] = []
    avg_y: list[float] = []

    for i in range(interval_count):
        if i in parameters.DISTANCE_TRANSMISSION_POWER_DICT and len(parameters.DISTANCE_TRANSMISSION_POWER_DICT[i]) > 0:
            values = sorted(parameters.DISTANCE_TRANSMISSION_POWER_DICT[i])  # We take the values sorted so that we can get the max and min values
            max_value = max(values)
            min_value = min(values)
            avg_value = round(sum(values) / len(values), 6)

            x_plot_pos = i*100 + 50 # The plotting position (horizontal) for all the points, which will be the middle of the interval

            for val in values:
                all_x.append(x_plot_pos)
                all_y.append(val)
            
            max_x.append(x_plot_pos); max_y.append(max_value)
            min_x.append(x_plot_pos); min_y.append(min_value)
            avg_x.append(x_plot_pos); avg_y.append(avg_value)
    
    # We create the image and plot the points
    fig, ax = plt.subplots(figsize=(10, 6)) # type: ignore

    if all_x:
        ax.scatter(all_x, all_y, color='gray', marker='o', label='TxPower') # type: ignore
    if max_x:
        ax.scatter(max_x, max_y, color='red', marker='o', label='Max TxPower') # type: ignore
    if min_x:
        ax.scatter(min_x, min_y, color='red', marker='o', label='Min TxPower') # type: ignore
    
    # We know create the lines between max and min in each column and between all the average values
    if min_x and max_x:
        ax.vlines(x=min_x, ymin=min_y, ymax=max_y, colors='black', linestyles='--', label='Min-Max') # type: ignore
    if avg_x:
        ax.plot(avg_x, avg_y, color='blue', marker='o', linestyle='-',label='Avg TxPower') # type: ignore

    ax.set_xlabel("Distance (in m)") # type: ignore
    ax.set_ylabel("Transmission Power (in Watt)") # type: ignore
    ax.set_title("Distance vs Transmission Power") # type: ignore
    ax.set_xticks(range(0, int(max_value_rounded)+1, 100)) # type: ignore
    ax.set_ylim(0, parameters.Transmitting_Power*1.1)
    ax.set_xlim(0, max_value_rounded)
    ax.grid(True, linestyle=':') # type: ignore
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left") # type: ignore

    fig.savefig(f"{parameters.DISTANCE_TRANSMISSON_POWER_FOLDER}{curr_sim_time}.png", bbox_inches='tight', dpi=250) # type: ignore
    plt.close(fig)
    return

# ANDRES: create a gif with all the images created beforehand
@profile
def createGif_Distance_TxPower():
    allFiles = [eachFile for eachFile in os.listdir(parameters.DISTANCE_TRANSMISSON_POWER_FOLDER) if eachFile.endswith('.png')]
    if allFiles:
        # Sort by numeric prefix
        allFiles.sort(key=lambda x: int(x.split('.')[0]))
        frames: list[Image.Image] = []
        for eachFile in allFiles:
            frames.append(Image.open(os.path.join(parameters.DISTANCE_TRANSMISSON_POWER_FOLDER, eachFile)))  
    
        # Convert to GIF and save
        output_path = os.path.join(parameters.DISTANCE_TRANSMISSON_POWER_FOLDER, "Distance-TransmissionPower_Graph.gif")
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=500, loop=0) # Set loop to 0 for an infinite loop
    return

# Sh: Flow source and destination nodes
@profile
def getFlowSrc(packetID: str) -> str: return packetID.split("_")[-3]
@profile
def getFlowDest(packetID: str) -> str: return packetID.split("_")[-2]

@profile
def getFlowDestStat(packetID: str) -> str:
    if not parameters.FORCED_ROUTES:
        return "Node" + str(parameters.Gateway_Node_ID)
    
    return packetID.split("_")[-2]

# Sh: Get number of times (in words) a packet is reinserted at the source node, and the remaining packet id
@profile
def getTimesPktReinserted(packetID: str):
    parts = packetID.split('_')
    rem_packetID = '_'.join(parts[:-1])
    return (parts[-1], rem_packetID)

# Sh: Returns node id without name
# AT: Don't need to use regex since nodeId_str is like Node123
@profile
def getNodeIdx(nodeId: str) -> int:
    return int(nodeId[4:])

# Sh: Compute transmission range to each neighbor based on their current interference levels. Assumption: curr node knows the stats at its neighbors.
@profile
def currTxRangesAndMaxCapacities(currNodeWithName: 'Node', allNodes: list['Node']) -> tuple[dict[int, float], dict[int, float]]:
    txRangesPerLink: dict[int, float] = {}
    linkCosts: dict[int, float] = {}
    currNode = getNodeIdx(currNodeWithName.name)

    for eachNghbrWithName in allNodes:
        eachNghbr = getNodeIdx(eachNghbrWithName.name)
        if(eachNghbr == currNode) or (eachNghbr == parameters.Gateway_Node_ID):  # Skip self
            continue

        # Default: If this neighbor has not received packet from any of its neighbors in the last interval
        interference_power = 0.0
        rcvd_power = 0.0

        # Case 1. currNode-eachNghbr link entry exists
        try:
            # This neighbor node has received packets from current node in last interval
            if (parameters.Link_Quality[eachNghbr][currNode]['RcvdPackets']['TotalPkts'] != 0):
                interference_power = parameters.Link_Quality[eachNghbr][currNode]['RcvdPackets']['AvgInterferingPower']
                # rcvd_power = parameters.Link_Quality[eachNghbr][currNode]['RcvdPackets']['AvgPowerRcvd'];
                # Sh: Do not use rcvd power from past packets as the node locations might have changed since then. Use power based on distance.
                
        # Case 2. currNode-eachNghbr link entry does not exist
        except:
            # This neighbor node has received packets from other nodes in last interval
            if (parameters.Link_Quality[eachNghbr]['InterferingPkts']['TotalPkts'] != 0):
                interference_power = parameters.Link_Quality[eachNghbr]['InterferingPkts']['AvgPower']
                
        # Default received power based on node locations
        distance = math.sqrt(pow(currNodeWithName.latitude - eachNghbrWithName.latitude,2) + pow(currNodeWithName.longitude - eachNghbrWithName.longitude,2)) + 1e-6 # To avoid distance=0
        # AT: 2/14/26 Use the power Hello packets will be transmitted at. For now, we are using power
        rcvd_power = currNodeWithName.power*(pow(parameters.WAVELENGTH/(4*pi*distance),2))
        # rcvd_power = currNodeWithName.tx_power_data*(pow(parameters.WAVELENGTH/(4*pi*distance),2))

        # Consider a node unreachable if the rcvd power < min power required to decode signal 
        if (rcvd_power < parameters.RADIO_SENSITIVITY):
            rcvd_power = 0.0

        # These tx ranges show the maximum distance signal can travel based on current interference situation at receiver node. Later, we will create edge if it is > distance between nodes.
        #txRangesPerLink[eachNghbr] = parameters.txRangeAtInterferenceLevel(parameters.Transmitting_Power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], interference_power)
        # ANDRES: changed to now use the routing power of the specific node + external interference of the receiving node
        txRangesPerLink[eachNghbr] = parameters.txRangeAtInterferenceLevel(currNodeWithName.routing_power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], eachNghbrWithName.external_interference)
        # Sh: If 2nd option used: Keep the topology based on GPS (helps in finding links for MCS0 index). However, change the link capacity.
        
        # max SINR for standard / max SINR supported at this link. Higher value corresponds to low margin in SINR improvement. // ANDRES: added external interference
        linkCosts[eachNghbr] = parameters.MAX_MIN_REQUIRED_SINR/(rcvd_power/(interference_power+parameters.NOISE_FLOOR + eachNghbrWithName.external_interference) + 1e-16)
        
    #print("at node: ", currNodeWithName.name, "tx ranges per link: ",txRangesPerLink,"\n link cost (higher -> less suitable):",linkCosts)
    return txRangesPerLink, linkCosts

_last_valid_line_cache: dict[str, int] = {}

@profile
def nodeLocation(nodeName: str, lineNo: int) -> tuple[float, float]:
    if not os.path.exists(parameters.TRAJECTORY_FOLDER):
        print(f"rf.py: Cannot find {parameters.TRAJECTORY_FOLDER}")
        return -1, -1

    file_path = (Path(parameters.STATIC_TOPOLOGY_PATH) / f"{nodeName}.txt")
    if not file_path.exists():
        print(f"rf.py: Cannot find {file_path}")
        return -1, -1

    file_key = file_path.as_posix()

    line = linecache.getline(file_key, lineNo)
    if not line.strip():
        if file_key in _last_valid_line_cache:
            lineNo = _last_valid_line_cache[file_key]
        else:
            with open(file_path) as f:
                for lineNo, _ in enumerate(f, 1):
                    pass
            _last_valid_line_cache[file_key] = lineNo

        line = linecache.getline(file_key, lineNo)

    new_location = line.split(",", 3)[:2]
    new_latitude = int(float(new_location[0]))
    new_longitude = int(float(new_location[1]))

    return (new_latitude, new_longitude)

@profile
def get_nodeEnergy(node: int, node_Energy: NDArray[np.float64]) -> np.float64 | Literal[0]:# returns minimum node Energy level in a given route
    return node_Energy[node] if node_Energy[node]>0 else 0

# Check route quality based on RLT and ETD, RE. # SD: Added Route Energy (RE) constraint
@profile
def hasHighRouteQuality(route: list[int], pktTTE: float, LinkStats: dict[tuple[int, int], float], node_Energy: NDArray[np.float64], routeSwitching: bool = False):
    # Initialization
    rlt = math.inf; etd = 0; re = math.inf
    pastWeightage = 0.2; currWeightage = 1-pastWeightage
    
    for hopNo in range(len(route)-1):  # Do not consider destination node in ETD calculation; Similarly Src, Dst are assumed to have constant energy levels        
        # 1. Get LLT for each link. RLT = min(LLT)
        link = (route[hopNo],route[hopNo+1]) if(int(route[hopNo]) < int(route[hopNo+1])) else (route[hopNo+1],route[hopNo])
        rlt = min(rlt,1e9*LinkStats[link])  # in nano seconds
        
        # 2. Get moving average of PST for each node -> Get ETD for the route
        etd += float(parameters.Node_Stats[route[hopNo]]['LastPST'])*pastWeightage + float(parameters.Node_Stats[route[hopNo]]['PST'])*currWeightage

        # 3. Get RouteEnergy (RE) = min nodeEnergy in route
        if route[hopNo] == parameters.Gateway_Node_ID:
            re = min(re, parameters.MOBILITY.MAX_INITIAL_ENERGY)
        else:
            re = min( re, get_nodeEnergy( route[hopNo], node_Energy ) )
    
    # High quality route if and only if (a) RLT is sufficient, AND (b) ETD is low.
    threshold = parameters.TH1 if (not routeSwitching) else parameters.TH2
    constraint_RLT = (rlt >= parameters.PACKET_TTL + parameters.DELTA)
    constraint_ETD = (pktTTE >= etd*threshold)
    constraint_RE = (re >= (parameters.MOBILITY.ENERGY_THRESHOLD + parameters.MOBILITY.MAX_ENERGY_DEPLETION_RATE * (parameters.PACKET_TTL + parameters.DELTA)/1e9)) 
    # RE >= 10.4 Energy level
    highQualityRoute = (constraint_RLT and constraint_ETD and constraint_RE)
    
    if parameters.PRINT_LOGS: 
      print("Route:", route, "RLT:", rlt/1e9, "Pkt TTE:", pktTTE, "ETD:", etd, "RE:", re, "RLT Constraint:", constraint_RLT, "ETD Constraint:", constraint_ETD, "RE Constraint",constraint_RE, "HQ Route?:", highQualityRoute)
    
    return highQualityRoute

# Check route quality based on RLT and ETD.
@profile
def hasHighRouteQuality_old(route: list[int], pktTTE: float, LinkStats: dict[tuple[int, int], float], routeSwitching: bool = False):
    # Initialization
    rlt = math.inf; etd = 0
    pastWeightage = 0.2; currWeightage = 1-pastWeightage
    
    for hopNo in range(len(route)-1):  # Do not consider destination node in ETD calculation
        # 1. Get LLT for each link. RLT = min(LLT)
        link = (route[hopNo],route[hopNo+1]) if(int(route[hopNo]) < int(route[hopNo+1])) else (route[hopNo+1],route[hopNo])
        rlt = min(rlt,1e9*LinkStats[link])  # in nano seconds
        
        # 2. Get moving average of PST for each node -> Get ETD for the route
        etd += float(parameters.Node_Stats[route[hopNo]]['LastPST'])*pastWeightage + float(parameters.Node_Stats[route[hopNo]]['PST'])*currWeightage
    
    # High quality route if and only if (a) RLT is sufficient, AND (b) ETD is low.
    threshold = parameters.TH1 if (not routeSwitching) else parameters.TH2
    constraint_RLT = (rlt >= parameters.PACKET_TTL + parameters.DELTA)
    constraint_ETD = (pktTTE >= etd*threshold)
    highQualityRoute = (constraint_RLT and constraint_ETD)
    
    if parameters.PRINT_LOGS: 
        print("Route:", route, "Pkt TTE:", pktTTE, "ETD:", etd, "RLT Constraint:", constraint_RLT, "ETD Constraint:", constraint_ETD, "HQ Route?:", highQualityRoute)

    return highQualityRoute


# Get suitable MCS index for each link on a route
@profile
def findSuitableMCS(route: list[int], wLC: dict[int, dict[int, float]]) -> list[int]:
    routeMCSIndex: list[int] = []
    for hopNo in range(len(route)-1):
        weighted_link_cost = max(wLC[route[hopNo]][route[hopNo+1]], wLC[route[hopNo+1]][route[hopNo]])
        curr_sinr = parameters.MAX_MIN_REQUIRED_SINR / weighted_link_cost    # Get link sinr
        # Lookup suitable MCS index; DEFAULT_DATA_MCS_INDEX is the minimum mcs required for data packet at each node
        mcs_index = parameters.DEFAULT_DATA_MCS_INDEX
        for eachEntry in parameters.MCS_SNR_TABLE:
            if (curr_sinr - 0.2 < parameters.MCS_SNR_TABLE[eachEntry]['MinSNR']): # Consider margin # SD: ??? was 0.5 in original version; changed to 0.2 CHECK!!!
                break
            mcs_index = eachEntry
        routeMCSIndex.append(mcs_index)
    return routeMCSIndex

# Get the remaining link lifetime for a link between nodes A and B at current time
@profile
def FindLLT(Id_A: str, Id_B: str, CurrTime: float):

    # Get trajectory number
    result = re.search(r'(\d+)GM_', parameters.MOBILITY_MODEL)
    if not result:
        raise ValueError("Couldn't get trajectory number")
    trajNo = int(result.group(1))
    #print("In FindLLT:",Id_A,Id_B,CurrTime,trajNo)
    
    dirPath = os.path.dirname(os.path.abspath(__file__))+"/nodeTrajectories/"
    filename = parameters.LLT_SAMPLE_FILE
    pll = -1.0
    #TrajectoryChange = 0   #0->false,1->true

    # Check if the file exists
    if(not os.path.exists(dirPath+filename)):
        print("Cannot locate file:",dirPath+filename)
        # Terminate the simulation
        exit(1)
    else:
        df: pd.DataFrame = pd.read_csv(dirPath+filename) # type: ignore

        # Find relavant entries
        df = df[ (df["RunNo"] == trajNo) & (((df["UAV_A"] == Id_A) & (df["UAV_B"] == Id_B)) | ((df["UAV_A"]== Id_B) & (df["UAV_B"] == Id_A))) ]; #print(df)
        
        if(len(df)==0):
            print(f"Could not find entry for {Id_A}-{Id_B} link in samples.dat! Potential reason: Link entry exists for < 3 Hello Interval (or 6 s) in standard OLSR. This wait is required to confirm link symmetry!")
            print("Use this link for a max of location update interval! Check condition: PLL < 0")
            pll = parameters.LOCATION_UPDATE_INTERVAL/1e9
            #exit(1)
        else:
            # Case 1: Get Sample where cretion time = current time
            entry = df[ df["SampleCreationTime"]==int(CurrTime) ]
            if(len(entry)==1):
                pll = (entry.LinkEndTime - entry.SampleCreationTime).values[0]; #print("Case 1: PLL", pll) # type: ignore
                #TrajectoryChange = 1

            else:
                # Case 2: Get Sample where creation time = current time+1
                entry = df[ df["SampleCreationTime"]==int(CurrTime)+1 ]
                if(len(entry)==1):
                    pll = (entry.LinkEndTime - entry.SampleCreationTime).values[0]; #print("Case 2: PLL", pll) # type: ignore
                    #TrajectoryChange = 1
                    
                # Case 3: A node may not be able to compute its Link Lifetime at the time of trajectory change or link establishment due to control packet collision.
                # This condition allows Link Lifetime computation at any intermediate timestamp.
                else:
                    entry = df [ (df["SampleCreationTime"]<int(CurrTime)) & (df["SampleEndTime"]>=int(CurrTime)) ]
                    if (len(entry)==1):
                        pll = (entry.LinkEndTime - int(CurrTime)).values[0]; #print("Case 3: PLL", pll) # type: ignore
                        #TrajectoryChange = 0
                        
        '''if (pll < 0):
            print(f"Link {Id_A}-{Id_B} has expired (PLL={pll}) but showing up in the network topology graph. Debug the issue!"); print(df)
            exit(1)'''
    
    return pll

# MC: Creates a reduced topology based on routes that are already being occupied
@profile
def removeNodesInTopology(route: list[int], topology: 'nx.Graph[int]', someNodes: list['Node'], time: int, neighbors: bool = True) -> tuple['nx.Graph[int]', list['Node']]:
    node_dict: dict[int, 'Node'] = {}
    all_node_ids: set[int] = set()
    for node in someNodes:
        nid = getNodeIdx(node.name)
        node_dict[nid] = node
        all_node_ids.add(nid)

    route_node_ids = set(route)

    # Protected nodes: source, destination, and their 1-hop neighbors
    source_id = route[0]
    destination_id = route[-1]
    protected_node_ids = {source_id, destination_id}

    source_neighbors: set[int] = set(topology.neighbors(source_id)) if topology.has_node(source_id) else set()
    dest_neighbors: set[int] = set(topology.neighbors(destination_id)) if topology.has_node(destination_id) else set()
    protected_node_ids.update(source_neighbors)
    protected_node_ids.update(dest_neighbors)

    if not neighbors:
        final_node_ids = all_node_ids - (route_node_ids - protected_node_ids)
    else:
        all_route_neighbors: set[int] = set()
        for route_node_id in route_node_ids:
            if route_node_id not in protected_node_ids and topology.has_node(route_node_id):
                all_route_neighbors.update(topology.adj[route_node_id])

        final_node_ids = all_node_ids - ((route_node_ids | all_route_neighbors) - protected_node_ids)

    filtered_nodes = [node_dict[nid] for nid in final_node_ids if nid in node_dict]

    # Create the reduced topology
    node_locations: dict[int, tuple[float, float]] = {}
    node_max_tx_ranges: dict[int, float] = {}
    node_curr_tx_ranges: dict[int, dict[int, float]] = {}  # Based on current interference levels at transmitter and reciever nodes
    weighted_link_capacities: dict[int, dict[int, float]] = {}  # Maximum achieveable sinr at current interference levels
    node_max_TRANSMISSION_ranges: dict[int, float] = {} # SD: nodes's actual transmission power's (NOT routing power/range) tx range

    for node in filtered_nodes:
        nid = getNodeIdx(node.name)
        node_locations[nid] = (node.latitude, node.longitude)
        # ANDRES: uses distance based on node routing power + external interference of the destination
        node_max_tx_ranges[nid] = parameters.txRangeAtInterferenceLevel(node.routing_power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], node.external_interference)
        # AT: 2/14/26 Use the power Hello packets will be transmitted at. For now, we are using power
        node_max_TRANSMISSION_ranges[nid] = parameters.txRangeAtInterferenceLevel(node.power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], node.external_interference)
        # node_max_TRANSMISSION_ranges[nid] = parameters.txRangeAtInterferenceLevel(node.tx_power_data, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], node.external_interference)

    for node in filtered_nodes:
        nid = getNodeIdx(node.name)
        node_curr_tx_ranges[nid], weighted_link_capacities[nid] = currTxRangesAndMaxCapacities(node, filtered_nodes)

    topology = createTopology(node_locations, node_max_tx_ranges, node_max_TRANSMISSION_ranges, node_curr_tx_ranges, weighted_link_capacities, time, False)

    return topology, filtered_nodes

# MC: Just to make sure nodes are in a route
@profile
def isInTopology(src: int, dest: int, nodes: list['Node']) -> bool:
  return src in [getNodeIdx(eachNode.name) for eachNode in nodes] and dest in [getNodeIdx(eachNode.name) for eachNode in nodes]

# MC: Determines whether there is any high quality route in a given topology
@profile
def containsHighRouteQuality(topology: 'nx.Graph[int]', src: int, dest: int, HOL_pkt_TTE: float, LLT_STATS: dict[tuple[int, int], float]):
  minLength: int = nx.shortest_path_length(topology, source=src, target=dest) # type: ignore
  allPath = nx.all_simple_paths(topology, source=src, target=dest, cutoff=minLength+parameters.HC_THRESHOLD)
  return any([hasHighRouteQuality(eachRoute, HOL_pkt_TTE, LLT_STATS, parameters.MOBILITY.NODE_ENERGY, True) for eachRoute in allPath])

# Get updated node location, network topology graph, routes and MCS index. Note: We use updated node location but 'OLDER' inetrference values (obtained during the previous inetrval) 
# to find edges between nodes. This may cause error since the revised SINR values at new node locations are not available.

@profile
def updateNetworkTopology(env: simpy.Environment, allNodes: list['Node']):
    # while True: #SHREY* Modified to call after running mobility at every time step 
    
    # if 1008 <= int(env.now/(1e9)) < 1015:
    #     parameters.PRINT_LOGS = True
    # else:
    #     parameters.PRINT_LOGS = False

    # print("AT nw-upd ", parameters.Node_Locations_from_MobilityModel)
    curr_sim_time = int(env.now/(1e9))
    node_locations: dict[int, tuple[float, float]] = {}
    node_max_tx_ranges: dict[int, float] = {}
    node_curr_tx_ranges: dict[int, dict[int, float]] = {}  # Based on current interference levels at transmitter and reciever nodes
    weighted_link_capacities: dict[int, dict[int, float]] = {}  # Maximum achieveable sinr at current interference levels
    node_max_TRANSMISSION_ranges: dict[int, float] = {} # SD: ???
    
    # Task 1. Update node locations
    for eachNode in allNodes:
        if int(eachNode.name[4:]) == parameters.Gateway_Node_ID:
            continue
        old_latitude = eachNode.latitude
        old_longitude = eachNode.longitude
        # Obtain new location in mobile scenarios
        if parameters.MOBILE_SCENARIO:
            # eachNode.latitude, eachNode.longitude = nodeLocation(eachNode.name,curr_sim_time+2)    # Start with 2nd line
            eachNode.latitude, eachNode.longitude = get_NodeLocation_from_MobilityModel(int(eachNode.name[4:])) #SHREY* get node positions at curr_sim_time from mobility model 
            # print("Node n location", eachNode.name, eachNode.latitude, eachNode.longitude, " at time ", curr_sim_time)
            if parameters.NODE_LOCATION_WAYPOINT_LOG_FLAG: #SHREY*: to log curr position and nxt-waypoint of node
                parameters.Node_Location_Waypoint[int(eachNode.name[4:]) , curr_sim_time - int(parameters.PKT_GENERATION_START_SEC)] = get_NodeLocation_NxtWaypoint_from_MobilityModel( int(eachNode.name[4:]) ) 

            # Update location at MAC layer
            eachNode.mac.latitude = eachNode.latitude
            eachNode.mac.longitude = eachNode.longitude
            # Update location at Phy layer
            eachNode.mac.phy.latitude = eachNode.latitude
            eachNode.mac.phy.longitude = eachNode.longitude
        if parameters.PRINT_LOGS:
            print("rf.py: At node %s time: %ss, Curr coordiantes: %f,%f, Updated coordinates: %f,%f" % (eachNode.name, curr_sim_time, old_latitude, old_longitude, eachNode.latitude, eachNode.longitude))
        
        #nodeId = int(re.search(r'\d+', eachNode.name).group()) # Id without name
        nodeId = getNodeIdx(eachNode.name)
        node_locations[nodeId] = (eachNode.latitude, eachNode.longitude)
        # ANDRES: uses distance based on node routing power + external interference of the destination
        node_max_tx_ranges[nodeId] = parameters.txRangeAtInterferenceLevel(eachNode.routing_power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], eachNode.external_interference)
        # AT: 2/14/26 Use the power Hello packets will be transmitted at. For now, we are using power
        node_max_TRANSMISSION_ranges[nodeId] = parameters.txRangeAtInterferenceLevel(eachNode.power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], eachNode.external_interference)
        # node_max_TRANSMISSION_ranges[nodeId] = parameters.txRangeAtInterferenceLevel(eachNode.tx_power_data, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], eachNode.external_interference)

        furthestDist = 0
        for otherNode in allNodes:
            if eachNode.id == otherNode.id: continue
            distance = ether.computeDistance(eachNode.latitude, eachNode.longitude, otherNode.latitude, otherNode.longitude)
            if distance <= parameters.MAXIMUM_ROUTING_RANGE:
                furthestDist = max(furthestDist, distance)
        parameters.DistToFurthestNodeInRange[eachNode.id] = furthestDist
    gateway_tx_range = parameters.txRangeAtInterferenceLevel(parameters.Transmitting_Power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'])
    node_max_tx_ranges[parameters.Gateway_Node_ID] = gateway_tx_range
    
    # Node locations have been updated

    # Get effective transmisison ranges and link cost
    for eachNode in allNodes:
        if int(eachNode.name[4:]) == parameters.Gateway_Node_ID:
            continue
        node_curr_tx_ranges[getNodeIdx(eachNode.name)], weighted_link_capacities[getNodeIdx(eachNode.name)] = currTxRangesAndMaxCapacities(eachNode, allNodes)
    
    gateway_curr_tx_ranges: dict[int, float] = {}
    gateway_weighted_link_capacities: dict[int, float] = {}
    for eachNode in allNodes:
        eachNodeIdx = getNodeIdx(eachNode.name)
        if eachNodeIdx == parameters.Gateway_Node_ID:
            continue
        gateway_curr_tx_ranges[eachNodeIdx] = gateway_tx_range
        node_curr_tx_ranges[eachNodeIdx][parameters.Gateway_Node_ID] = gateway_tx_range
        gateway_weighted_link_capacities[eachNodeIdx] = 1e-12
        weighted_link_capacities[eachNodeIdx][parameters.Gateway_Node_ID] = 1e-12
    node_curr_tx_ranges[parameters.Gateway_Node_ID] = gateway_curr_tx_ranges
    weighted_link_capacities[parameters.Gateway_Node_ID] = gateway_weighted_link_capacities

    # Debug purposes
    # print("Time", env.now)
    # print("Link quality:",parameters.Link_Quality)
    # print("Node Stats:",parameters.Node_Stats)
    # print("Curr tx ranges:",node_curr_tx_ranges)
    # print("Curr link capacities:",weighted_link_capacities)
    # print("Node's max tx ranges:", node_max_tx_ranges)
    
    # ANDRES: create the image for the current time and store it in the folder. We skip second 0 because there have been no packages sent yet
    if curr_sim_time != 0:
        plot_distance_txPower_graph(curr_sim_time)

    # Task 2. Create network graph, get updated link lifetime (LLT) values and visualize it
    topology = createTopology(node_locations, node_max_tx_ranges, node_max_TRANSMISSION_ranges, node_curr_tx_ranges, weighted_link_capacities, curr_sim_time)
    parameters.Current_Topology = copy.deepcopy(topology.copy(as_view=True))

    # print("TOPO=", topology.graph)


    # Get LLT values
    LLT_STATS: dict[tuple[int, int], float] = {}
    for edge in topology.edges():
        node1, node2 = edge
        if node1 == parameters.Gateway_Node_ID or node2 == parameters.Gateway_Node_ID:
            LLT_STATS[edge] = parameters.SIM_TIME
        # LLT_STATS[edge] = FindLLT (node1,node2,curr_sim_time) if parameters.MOBILE_SCENARIO else parameters.SIM_TIME
        else:
            LLT_STATS[edge] = FindLLT_shreyas (node1,node2, node_max_tx_ranges, int( (parameters.PACKET_TTL + parameters.DELTA)/1e9) ) if parameters.MOBILE_SCENARIO else parameters.SIM_TIME #SHREY*
    # print("At time:",curr_sim_time,"Link Lifetime:",LLT_STATS)
    
    route_details = parameters.Route_Details.copy()
    if parameters.PRIORITY_BASED_ROUTING:
        route_details = dict(sorted(route_details.items(), key=lambda item: item[1]['Priority'], reverse=True))
    
    # MC: Case 1 and 2 map topologies
    currTopology = topology.copy()
    someNodes = allNodes.copy()
    gateway_name = "Node" + str(parameters.Gateway_Node_ID)
    for i in range(len(someNodes)):
        if someNodes[i].name == gateway_name:
            someNodes.pop(i)
            break
    currTopology2 = topology.copy()
    mostNodes = someNodes.copy()

    # Task 3. Update routes
    # Case A. Check the quality of current route. If it is below threshold, find a new route.
    # Case B. If no high-quality route is available, source node should select the best sub-optimal route.
    # Case C. If no route is avaiable, source node directly tries to reach the destination node. However, update route validity to false.
    for routeNo in route_details:
        eachRoute = parameters.Route_Details[routeNo]['Route']
        # print("rf.py: Before route update: Route No.:%d, Route: %s" % (routeNo, eachRoute))
        src = eachRoute[0]
        
        if parameters.FORCED_ROUTES:
            dest = eachRoute[-1]
        elif parameters.MOBILITY.ROUTING_COST_TYPE == 3:
            dest = min(parameters.BSIDs, key=lambda id: ether.computeDistance(*parameters.Route_Details[routeNo]['Target'], *node_locations[id])) # Get the closest base station to the source node
        else:
            dest = parameters.Gateway_Node_ID

        # Get TTE of the HOL packet at the source node
        HOL_pkt_TTE = parameters.PACKET_TTL/1e9 # Initialization
        for eachNode in allNodes:
            if (getNodeIdx(eachNode.name) == src):
                HOL_pkt_TTE = eachNode.getHOLPktTTE(curr_sim_time)
                break
        
        try:
            # Sh: Recompute a new route only if the current route does not exist anymore or its quality is below required threshold
            if ((parameters.PRIORITY_BASED_ROUTING and (not isInTopology(src, dest, someNodes) or (not nx.is_path(currTopology, eachRoute) or not hasHighRouteQuality(eachRoute, HOL_pkt_TTE, LLT_STATS, parameters.MOBILITY.NODE_ENERGY, True))))
                or (not parameters.PRIORITY_BASED_ROUTING and ((not nx.is_path(topology, eachRoute)) or (not hasHighRouteQuality(eachRoute, HOL_pkt_TTE, LLT_STATS, parameters.MOBILITY.NODE_ENERGY, True))))):
                
                minLength: int
                allPaths: Iterable[list[int]]
                # Get all the routes up to a length = shortestHC + HC Threshold
                # MC: Three-step routing priority scheme
                if (parameters.PRIORITY_BASED_ROUTING) and (not isInTopology(src, dest, someNodes) or not nx.has_path(currTopology,src,dest)): #or not containsHighRouteQuality(currTopology, src, dest, HOL_pkt_TTE,LLT_STATS):
                    if (not isInTopology(src, dest, mostNodes) or not nx.has_path(currTopology2, src, dest)) or not containsHighRouteQuality(currTopology2, src, dest, HOL_pkt_TTE, LLT_STATS):
                        # Case 3: Entire map is considered
                        print('Route not available with reduced topology, layer 3')
                        minLength = nx.shortest_path_length(topology, source=src, target=dest) # type: ignore
                        allPaths = nx.all_simple_paths(topology, source=src, target=dest, cutoff=minLength+parameters.HC_THRESHOLD)
                    else: 
                        # Case 2: All unreserved nodes in map if high-priority flow only reserves route nodes
                        print('Route not available with reduced topology, layer 2')
                        minLength = nx.shortest_path_length(currTopology2, source=src, target=dest) # type: ignore
                        allPaths = nx.all_simple_paths(currTopology2, source=src, target=dest, cutoff=minLength+parameters.HC_THRESHOLD)
                else:
                    # Case 1: All unreserved nodes in map if high-priority flow reserves route nodes and 1-hop neighbors
                    # Get all the routes up to a length = shortestHC + HC Threshold
                    minLength = nx.shortest_path_length(currTopology, source=src, target=dest) # type: ignore
                    allPaths = nx.all_simple_paths(currTopology, source=src, target=dest, cutoff=minLength+parameters.HC_THRESHOLD)

                # Compute HC, IL, Route Capacity (i.e., max data rate supported), Route quality meeting threshold, is route tangled
                @profile
                def process_route(route: list[int]) -> tuple[list[int], int, int, float, bool, bool]:
                    # Remove Gateway node
                    if route[-1] == parameters.Gateway_Node_ID:
                        route = route[:-1]
                    
                    # 1. Hop count
                    rHC = len(route)-1
                    
                    # 2. Total Interfering links
                    rIL = 0
                    for hopNo in range(len(route)):
                        nodeIntfLinks = parameters.Node_Stats[route[hopNo]]['IL']
                        rIL += len(nodeIntfLinks)
                        # At any intermediate node, upstream and downstream links will interfere with each other.
                        # Upstream link at intermediate and destination nodes
                        if ((hopNo>0) and (f'{route[hopNo-1]}-{route[hopNo]}' not in nodeIntfLinks)): rIL += 1
                        # Downstream link at source and intermediate nodes
                        if ((hopNo!=len(route)-1) and (f'{route[hopNo]}-{route[hopNo+1]}' not in nodeIntfLinks)): rIL += 1

                    # 3. Max data rate supported on the route will depend on the minimum MCS index supported by all of its link
                    # Step 1: Get MCS index for each link on this route
                    routeMCSIndecies = findSuitableMCS(route, weighted_link_capacities)
                    # Step 2: Find data rate corresponding to the minimum MCS index supported
                    rDR = parameters.MCS_SNR_TABLE[min(routeMCSIndecies)]['DataRate']

                    # 4. HQ route check
                    isHQRoute = hasHighRouteQuality(route,HOL_pkt_TTE, LLT_STATS, parameters.MOBILITY.NODE_ENERGY)

                    # 5. Is route tangled?
                    '''
                    Simple path does not repeat a node. However a path may be tangled. For example, consider a 4-node network topology. Node 1 has links with nodes 2 and 3. Remaining nodes 2, 3, 4
                    have links with each other. All simple paths between nodes 1 and 4 will be: [1,2,4], [1,3,4], [1,2,3,4], [1,3,2,4]. Here, paths [1,2,3,4] and [1,3,2,4] are tangled, and if used
                    will increase the channel contention and thereby intra-flow interference and number of packet retransmissions. 
                    Therefore, we flag such tangled routes and do not consider them in route selection.
                    '''
                    isTangledRoute = False
                    for ii in range(len(route)-1):
                        for jj in range(ii + 2, len(route)):
                                if topology.has_edge(route[ii], route[jj]):
                                    isTangledRoute = True
                                    break

                    #print(route,rHC,rIL,rDR,isHQRoute,isTangledRoute)
                    return route, rHC, rIL, rDR, isHQRoute, isTangledRoute

                # Process each path in parallel
                processedAllPaths: list[ProcessedPathEntry] = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    pathStats = [executor.submit(process_route, eachPath) for eachPath in allPaths]

                    for pathStat in as_completed(pathStats):
                        path, pHC, pIL, pDR, hqPath, isTangled = pathStat.result()
                        #print("Processed Route:", path, "HC:", pHC, "IL:", pIL, "Max Data Rate Supported:", pDR, "HQ path?:", hqPath, "tangled route?:", isTangled)
                        
                        # Remove tangled paths
                        if not isTangled:
                            processedAllPaths.append({'Path': path, 
                                                    'HC': pHC, 
                                                    'IL': pIL, 
                                                    'Max Data Rate Supported': max(pDR,parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['DataRate']), # At least support min data rate
                                                    'HQ Path': hqPath,
                                                    'Tangled Path': isTangled,
                                                    'Cost': float("inf")
                                                    })

                # Get high-quality paths. If no high-quality path, search from sub-optimal paths.
                hq_processedPaths: list[ProcessedPathEntry] = []
                
                min_HC, hqmin_HC = float("inf"), float("inf") # To avoid divide by 0 problem
                min_IL, hqmin_IL = float("inf"), float("inf") # To avoid divide by 0 problem
                max_data_rate, hqmax_data_rate = 0, 0  # Will not be less than data rate supported at the default index
                
                # Find the min HC, min IL, max route capacity (in data rate)
                for path in processedAllPaths:
                    if path['HC'] < min_HC:
                        min_HC = path['HC']
                    if path['IL'] < min_IL:
                        min_IL = path['IL']
                    if path['Max Data Rate Supported'] > max_data_rate:
                        max_data_rate = path['Max Data Rate Supported']
                    if path['HQ Path'] == 1:
                        hq_processedPaths.append(path)
                        if path['HC'] < hqmin_HC:
                            hqmin_HC = path['HC']
                        if path['IL'] < hqmin_IL:
                            hqmin_IL = path['IL']
                        if path['Max Data Rate Supported'] > hqmax_data_rate:
                            hqmax_data_rate = path['Max Data Rate Supported']
                
                if len(hq_processedPaths) > 0:
                    pathsToConsider = hq_processedPaths
                    min_HC = hqmin_HC
                    min_IL = hqmin_IL
                    max_data_rate = hqmax_data_rate
                else:
                    pathsToConsider = processedAllPaths
                
                min_HC = max(1, min_HC)
                min_IL = max(1, min_IL)
                
                # Get path cost = [w1*(HC_R/min_HC) + w2*(IL_R/(min_IL + alpha*IL_R))]*(max data rate of all routes/max data rate on path R)
                for path in pathsToConsider:
                    if parameters.MOBILITY.ROUTING_COST_TYPE == 1:
                        path['Cost'] = (parameters.W1 * (path['HC'] / min_HC) + parameters.W2 * (path['IL'] / (min_IL + parameters.ALPHA * path['IL']))) * max_data_rate / path['Max Data Rate Supported']
                    elif parameters.MOBILITY.ROUTING_COST_TYPE == 2:
                        path['Cost'] = len(path['Path'])
                    elif parameters.MOBILITY.ROUTING_COST_TYPE == 3:
                        path['Cost'] = len(path['Path'])

                if True:
                    keys = pathsToConsider[0].keys()
                    print(" | ".join(keys))
                    print("-" * 30)
                    for row in pathsToConsider:
                        print(" | ".join(str(row[key]) for key in keys)) # type: ignore[reportUnknownArgumentType]

                # Use the least cost route
                if parameters.STRICT_ROUTE_SELECTION:
                    foundPaths: list[ProcessedPathEntry] = [pathsToConsider[0]]
                    cur_cost = foundPaths[0]['Cost']
                    for i in range(1, len(pathsToConsider)):
                        if pathsToConsider[i]['Cost'] < cur_cost:
                            cur_cost = pathsToConsider[i]['Cost']
                            foundPaths = [pathsToConsider[i]]
                        elif pathsToConsider[i]['Cost'] == cur_cost:
                            foundPaths.append(pathsToConsider[i])
                
                    foundPaths.sort(key=lambda x: (len(x['Path']), x['Path']))
                    foundPath = foundPaths[0]
                else:
                    foundPath = pathsToConsider[0]
                    cur_cost = foundPath['Cost']
                    for i in range(1, len(pathsToConsider)):
                        if pathsToConsider[i]['Cost'] < cur_cost:
                            cur_cost = pathsToConsider[i]['Cost']
                            foundPath = pathsToConsider[i]
                
                print("Found Route:\n",foundPath)
                
                parameters.Route_Details[routeNo]['Route'] = foundPath['Path']
                parameters.Route_Details[routeNo]['HQRoute'] = foundPath['HQ Path'] == 1
                parameters.Route_Details[routeNo]['ValidRoute'] = True
                parameters.Route_Details[routeNo]['InterferingLinks'] = foundPath['IL']
                
            # else, keep using the current route. Its valid and has high route quality.
            else: 
                parameters.Route_Details[routeNo]['HQRoute'] = True
                parameters.Route_Details[routeNo]['ValidRoute'] = True
                if 'InterferingLinks' not in parameters.Route_Details[routeNo]:
                    # Calculate InterferingLinks for the existing route if it's kept and InterferingLinks wasn't previously stored or needs update
                    # This would involve calling a part of the process_route logic for the current eachRoute
                    # For simplicity here, we'll set it to 0 if not found, but ideally, it should be accurate.
                    # temp_route_hc, temp_route_il, _, _, _ = process_route_basic_stats(eachRoute, topology, LLT_STATS, HOL_pkt_TTE, weighted_link_capacities)
                    # parameters.Route_Details[routeNo]['InterferingLinks'] = temp_route_il
                    parameters.Route_Details[routeNo]['InterferingLinks'] = parameters.Route_Details[routeNo].get('InterferingLinks', 0) # Keep existing or default to 0
        except nx.NetworkXNoPath:
            if parameters.PRINT_LOGS:
                print("rf.py: At time: %s, No route from %s to %s" % (curr_sim_time, src, dest))
            parameters.Route_Details[routeNo]['Route'] = [src,dest]
            parameters.Route_Details[routeNo]['ValidRoute'] = False
            parameters.Route_Details[routeNo]['HQRoute'] = False
            parameters.Route_Details[routeNo]['IL'] = 0 # Or some indicator for no valid route
        #print("rf.py: After route update: Route No.:%d, Route: %s" % (routeNo, parameters.Route_Details[routeNo]))
        
        # Sh: If enabled, discard the packets from the intermediate nodes of the previous route as they no longer have a route to destination node, and reinsert these packets at the source node.
        if parameters.ENABLE_PACKET_REINSERTION:
            newRoute = parameters.Route_Details[routeNo]['Route']
            if eachRoute != newRoute:
                outdatedNodes = list(set(eachRoute)-set(newRoute))
                if parameters.PRINT_LOGS:
                    print("In rf.py: Old route:", eachRoute, "New route:", newRoute, "Obsolete nodes:", outdatedNodes)
                tl.ReinsertPacketsAtFlowSrc(f"Node{eachRoute[0]}_Node{eachRoute[-1]}",outdatedNodes,allNodes)

        # Task 4. Update MCS index for each node on the route
        if parameters.ENABLE_MCS_SNR:
            updatedMCSIndex = findSuitableMCS(parameters.Route_Details[routeNo]['Route'],weighted_link_capacities)
            parameters.Route_Details[routeNo]['MCS_Index'] = updatedMCSIndex
        else: # Without adaptive MCS-SNR
            updateRouteMCS(routeNo, force=True)

        # SD:
        # Task 4.5. Update RouteNodeDensity:
        if parameters.MOBILITY.TC_PIPE_mobility_ON:
            # G_now = nx.from_numpy_array(connMat)
            routes_node_density = getRouteNodeDensity(topology, routeNo) # SD: Updates value in parameters.Route_Details[routeNo]['RouteNodeDensity']
            # print("routes_node_density", routes_node_density)
            parameters.Route_Details[routeNo]['RouteNodeDensity'] = routes_node_density
            # print("Route:", parameters.Route_Details[routeNo]['Route'], "-> Routes_node_density:", routes_node_density)

        # MC: Reduces the size of the available topology based on the nodes being occupied by current route      
        if parameters.PRIORITY_BASED_ROUTING:
            if parameters.Route_Details[routeNo]['Priority'] >= parameters.HIGH_PRIORITY:
                currTopology, someNodes = removeNodesInTopology(parameters.Route_Details[routeNo]['Route'], currTopology, someNodes, curr_sim_time)
                print('Remaining Nodes Available, neighbors considered: ' + str([getNodeIdx(eachNode.name) for eachNode in someNodes]))

                currTopology2, mostNodes = removeNodesInTopology(parameters.Route_Details[routeNo]['Route'], currTopology2, mostNodes, curr_sim_time, False)
                print('Remaining Nodes Available, neighbors not considered: ' + str([getNodeIdx(eachNode.name) for eachNode in mostNodes]))
      

    print("\nrf.py: Time:",curr_sim_time, "Route details:", parameters.Route_Details)
    
    # Task 5. Clear link qualities, node statistics
    parameters.initializeLinkQuality(curr_sim_time)  # in second
    parameters.initializeNodeStats(curr_sim_time)  # in second
    
    # DONE: 10/23 Clear interference tables
    for i in range(len(parameters.INTF_Registry)):
        parameters.INTF_Registry[i] = None
    
    # Task 6. Delete outdated NAV entries at all nodes
    parameters.deleteAllOldNAVEntries(round(env.now)) # in ns
    if parameters.PRINT_LOGS: print("rf.py: Remove obsolete NAV entires from entire table: NAV Table:",parameters.NAV_Table,"\n")

    # Wait until next update
    yield env.timeout(parameters.LOCATION_UPDATE_INTERVAL)
    
UpdateRouteCalls: int = 0

@profile
def updateRouteMCS(routeIdx: Optional[int] = None, force: bool = False):
    global UpdateRouteCalls
    print(f"Called Update: {UpdateRouteCalls = }, {force = }", flush=True)
    
    if not force:
        UpdateRouteCalls += 1
        if UpdateRouteCalls != len(parameters.NODE_REGISTRY):
            return
        UpdateRouteCalls = 0
    print("!! UPDATING !!", flush=True)
    routes = list(parameters.Route_Details.keys()) if routeIdx is None else [routeIdx]
    for routeNo in routes:
        # AT: Keep DEFAULT_DATA_MCS_INDEX; this is the minimum mcs index required for the data packet at each node
        updatedMCSIndex = [parameters.DEFAULT_DATA_MCS_INDEX for _ in range(len(parameters.Route_Details[routeNo]['Route'])-1)]  # Reset to default
        # Update MCS index for high power nodes in this route
        for i in range(len(updatedMCSIndex)):
             each_node = parameters.NODE_REGISTRY[f"Node{parameters.Route_Details[routeNo]['Route'][i]}"]
             updatedMCSIndex[i] = each_node.operating_mcs
        
        if routeNo in parameters.FORCED_ROUTE_MCS:
            route_node_mcs_map = parameters.FORCED_ROUTE_MCS[routeNo]
            for i in range(len(updatedMCSIndex)):
                if parameters.Route_Details[routeNo]['Route'][i] in route_node_mcs_map:
                    updatedMCSIndex[i] = route_node_mcs_map[parameters.Route_Details[routeNo]['Route'][i]]

        # for i in range(len(updatedMCSIndex)):
        #     each_node = parameters.NODE_REGISTRY[f"Node{parameters.Route_Details[routeNo]['Route'][i]}"]
            # each_node.operating_mcs = updatedMCSIndex[i]
            
        print("routeNo, updatedMCSIndex = ", routeNo, updatedMCSIndex)
        parameters.Route_Details[routeNo]['MCS_Index'] = updatedMCSIndex

# Get the node denisty of a given route
@profile
def getRouteNodeDensity(G: 'nx.Graph[int]', routeNo: int):
    path = parameters.Route_Details[routeNo]['Route']   # route
    path_valid = parameters.Route_Details[routeNo]['ValidRoute'] # route_validity
    node_density: list[Optional[int]] = [] # node density of node in route
    if path_valid: 
        for i,node in enumerate(path):
            # get no . of neighbors after disgarding upstream and ownstrean nodes
            if i == 0 or i == len(path)-1:
                node_density.append(nx.degree(G, node) - 1) # type: ignore
            else:
                node_density.append(nx.degree(G,node) -2) # type: ignore
    else:
        node_density = [None, None] # return densty = None if route broken
    return node_density

# Sh: Find route for a given source destination pair
@profile
def getRoute(flowSrc: str, flowDst: str) -> list[int]:
    foundRoute = []
    route_details = parameters.Route_Details
    if parameters.PRIORITY_BASED_ROUTING:
        route_details = parameters.Route_Details.copy()
        route_details = dict(sorted(route_details.items(), key=lambda item: item[1]['Priority'], reverse=True))
    
    for eachRouteIdx in route_details:
        eachRoute = parameters.Route_Details[eachRouteIdx]['Route']
        if ((flowSrc == "Node"+str(eachRoute[0])) & (flowDst == "Node"+str(eachRoute[-1]))):
            foundRoute = eachRoute
            break
    return foundRoute

# AADITYA: getRoute was doing two things using a flag, split checkRouteValidity into its own function
@profile
def checkRouteValidity(flowSrc: str, flowDst: str) -> bool:
    validRoute = False
    route_details = parameters.Route_Details
    if parameters.PRIORITY_BASED_ROUTING:
        route_details = parameters.Route_Details.copy()
        route_details = dict(sorted(route_details.items(), key=lambda item: item[1]['Priority'], reverse=True))

    for eachRouteIdx in route_details:
        eachRoute = parameters.Route_Details[eachRouteIdx]['Route']
        if ((flowSrc == "Node"+str(eachRoute[0])) & (flowDst == "Node"+str(eachRoute[-1]))):
            validRoute = parameters.Route_Details[eachRouteIdx]['ValidRoute']
            break
    return validRoute

#-----------------------------------------------------------------------------------------
#SHREY*
@profile
def get_NodeLocation_from_MobilityModel(node: int) -> tuple[np.float64, np.float64]:
    # if node == parameters.Gateway_Node_ID:
    #     return parameters.GATEWAY_LOCATION
    # New node location from mobility model at current time it is called
    new_latitude, new_longitude = parameters.Node_Locations_from_MobilityModel[node]
    return new_latitude, new_longitude

@profile
def get_NodeLocation_NxtWaypoint_from_MobilityModel(node: int) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    # if node == parameters.Gateway_Node_ID:
    #     return parameters.GATEWAY_LOCATION + parameters.GATEWAY_LOCATION
    # New node location and next-waypoint from mobility model at current time it is called
    new_latitude, new_longitude = parameters.Node_Locations_from_MobilityModel[node]
    nextwaypoint_new_latitude, nextwaypoint_new_longitude = parameters.Node_NextWaypoints_from_MobilityModel[node]
    # print([new_latitude, new_longitude, nextwaypoint_new_latitude, nextwaypoint_new_longitude])
    return new_latitude, new_longitude, nextwaypoint_new_latitude, nextwaypoint_new_longitude


# Get the remaining link lifetime for a link between nodes A and B at current time
@profile
def FindLLT_shreyas(Id_A: int, Id_B: int, max_transmission_ranges: dict[int, float], PACKET_TTL_plus_DELTA: int):
    """
    PACKET_TTL_plus_DELTA (seconds) ; input used = int( (parameters.PACKET_TTL + parameters.DELTA)/1e9) = (int(3+1)) = 4s

    Return:
    LLT (seconds)
    """
    # return float('inf')
    A0 = np.array( parameters.Node_Locations_from_MobilityModel[Id_A] )
    B0 = np.array( parameters.Node_Locations_from_MobilityModel[Id_B] )

    dif_vector_A = np.array( parameters.Node_NextWaypoints_from_MobilityModel[Id_A]) - A0
    norm_A = np.linalg.norm(dif_vector_A)
    if norm_A == 0:
        direction_unitvector_A = np.zeros_like(dif_vector_A)  # Node A is stationary
    else:
        direction_unitvector_A = dif_vector_A / norm_A

    dif_vector_B = np.array( parameters.Node_NextWaypoints_from_MobilityModel[Id_B]) - B0
    norm_B = np.linalg.norm(dif_vector_B)
    if norm_B == 0:
        direction_unitvector_B = np.zeros_like(dif_vector_B)  # Node B is stationary
    else:
        direction_unitvector_B = dif_vector_B / norm_B

    llt=0
    for t in range(0, PACKET_TTL_plus_DELTA+1 + 1):  # check LLT until till ((packet TTL + delta) + 1) seconds = 4+1 =5 s
        At = A0 + direction_unitvector_A* parameters.NODE_SPEED * t
        Bt = B0 + direction_unitvector_B* parameters.NODE_SPEED * t
        
        distance = np.linalg.norm(At - Bt)

        if distance <= min(max_transmission_ranges[Id_A], max_transmission_ranges[Id_B]):
            llt = t
        else:
            break
    # print(Id_A, Id_B, " LLT= ", llt, distance, At, Bt, A0, B0)
    return llt
