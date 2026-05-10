# === AUTO-GENERATED FROM CONFIG START ===
from typing import TypedDict
from types import SimpleNamespace
import numpy as np
from numpy.typing import NDArray
MCSTableEntry = TypedDict("MCSTableEntry", {"Modulation": str, "Coding": float, "MinSNR(dB)": int, "MinSNR": float, "DataRate": float})
RESULTS_FOLDER: str
NETWORK_GRAPHS: str
PHEROMONE_GRAPHS: str
TRAJECTORY_FOLDER: str
DISTANCE_TRANSMISSON_POWER_FOLDER: str
PRINT_LOGS: bool
PRINT_LOG_TOPOLOGY_METRICS: bool
SNAPSHOT_CODE: bool
ENABLE_MCS_SNR: bool
CHANNEL_AWARE_ROUTE_COMPUTATION: bool
ENABLE_PACKET_REINSERTION: bool
MOBILE_SCENARIO: bool
STATIC_TOPOLOGY_PATH: str
FORCED_ROUTES: list[list[int]]
FORCED_ROUTE_MCS: dict[int, dict[int, int]]
EXT_INTERFERENCE: dict[int, float]
ENABLE_Q_MNGMNT: bool
USE_FLOW_PRIORITY: bool
STRICT_ROUTE_SELECTION: bool
QUEUE_MANAGEMENT_PLUS: bool
CSMA_CA_BACKOFF: bool
PRIORITY_BASED_ROUTING: bool
ENABLE_POWER_CONTROL: bool
ENABLE_CONGESTION_DETECTION: bool
CONGESTION_BASED_MCS: bool
ROUTES_WITH_CONGESTION_BASED_MCS: list[int]
CONGESTION_BASED_MCS_INCREMENT_THRESHOLD: float
CONGESTION_BASED_MCS_DECREMENT_THRESHOLD: float
CONGESTION_BASED_MCS_INCREMENT_STEP: int
CONGESTION_BASED_MCS_DECREMENT_STEP: int
ENABLE_INDIVIDUAL_PKT_SINR: bool
MAX_MAC_PAYLOAD_LENGTH: int
BASE_PAYLOAD_LENGTH: int
USE_PACKET_AGGREGATION: bool
PACKET_LOSS_RATE: float
PAYLOAD_DATA_RATE: list[float]
TARGET_PKT_GENERATION_RATE: list[float]
PACKET_INTER_ARRIVAL_TIME: list[float]
PACKET_TTL: float
MOBILITY_MODEL: str
AREA_X: float
AREA_Y: float
NUMBER_OF_BS: int
NUMBER_OF_NODES: int
MAX_QUEUE_SIZE: int
NODE_SPEED: float
LLT_SAMPLE_FILE: str
PKT_GENERATION_START_SEC: float
PKT_GENERATION_END_SEC: float
PKT_GENERATION_START_TIME: float
PKT_GENERATION_END_TIME: float
SIM_TIME: float
PKT_FORWARDED_SUCCESSFULLY: int
PKT_DROPPED: int
DROP_REASON_OVERFLOW: str
DROP_REASON_EXPIRY: str
DROP_REASON_Q_MNGMNT: str
DROP_REASON_RETRYLIMIT: str
DROP_REASON_NO_ROUTE: str
WAITING_FOR_TX: str
MCS_SNR_TABLE: dict[int, MCSTableEntry]
DEFAULT_DATA_MCS_INDEX: int
MAX_DATA_MCS_INDEX: int
CONTROL_MCS_INDEX: int
CHANNEL_MHz: float
FREQUENCY: float
WAVELENGTH: float
RADIO_SWITCHING_TIME: float
STEP_SIZE: float
RADIO_SENSITIVITY: float
NOISE_FLOOR: float
MIN_MIN_REQUIRED_SINR: float
MAX_MIN_REQUIRED_SINR: float
SLOT_DURATION: float
SIFS_DURATION: float
DIFS_DURATION: float
RTS_LENGTH: int
CTS_LENGTH: int
MAC_HEADER_LENGTH: int
PHY_HEADER_LENGTH: int
ACK_LENGTH: int
RTS_THRESHOLD: int
USE_RTS_CTS: bool
CW_MIN: int
CW_MAX: int
MAX_RETRY_LIMIT: int
LOCATION_UPDATE_INTERVAL: float
QUEUE_UPDATE_INTERVAL: float
TCP_ENABLED: bool
DELTA: float
TH1: float
TH2: float
ALPHA: float
W1: float
W2: float
HC_THRESHOLD: int
SUR_THRESHOLD: float
FIRST_N_PKTS: int
NODE_LOCATION_WAYPOINT_LOG_FLAG: bool
class Mobility(SimpleNamespace):
	TC_PIPE_mobility_ON: bool
	NODE_FAILURE_PERCENTAGE: float
	MAX_INITIAL_ENERGY: float
	NUM_FAILURES: int
	NUM_ACTIVE: int
	ENERGY_NODE_FAILURE_VALUE: float
	ENERGY_THRESHOLD: float
	ENERGY_RECHARGE_LEVEL: float
	MAX_ENERGY_DEPLETION_RATE: float
	ENERGY_DEPLETION_RATE: np.typing.NDArray[np.number]
	ROUTING_COST_TYPE: int
	TARGET_LOCATION: list[list[int]]
	BS_SCHEME: list[tuple[float, float]]
MOBILITY: Mobility
# === AUTO-GENERATED FROM CONFIG END ===

import sys
import atexit
from scipy.constants import c, pi
import math
import os
import networkx as nx
import load_config
import simpy
import shutil
from pathlib import Path
import datetime
import json
import subprocess
import zipfile
from collections import defaultdict
from functools import lru_cache
import phy

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, NotRequired, Literal, Protocol, List, Tuple, Set, overload, cast

if TYPE_CHECKING:
    from node import Node

from line_profiler import profile # type: ignore[reportAssignmentType]

try:
	@profile
	def check_for_line_profiler() -> None:
		pass
except:
	def profile(f: Callable[..., Any]) -> Callable[..., Any]:
		return f


class MobilityExt(Mobility):
    NODE_ENERGY: NDArray[np.float64]

MOBILITY = MobilityExt() # type: ignore[reportConstantRedefinition]

INTF_Registry: List[Optional[float]]
Routes: List[Tuple[List[int], tuple[float, float]]]
Transmitting_Power: float
Maximum_Transmitting_Power: float
BSIDs: List[int] = []
Gateway_Node_ID: int
MAX_Transmission_Range: float
MAX_Transmission_Range_Squared: float

RouteDetailsEntry = TypedDict("RouteDetailsEntry", {'Route': List[int], 'MCS_Index': List[int], 'Priority': int, 'ValidRoute': bool, 
								'HQRoute': bool, 'RouteNodeDensity': List[Optional[int]] | List[None], 'InterferingLinks': int, 'Target': tuple[float, float], 'IL': NotRequired[int]})
Route_Details: dict[int, RouteDetailsEntry] = {}

InterferingPktsEntry = TypedDict('InterferingPktsEntry', {'TotalPkts': int,'AvgPower': float})
LinkQualityRcvdPktsEntry = TypedDict("LinkQualityRcvdPktsEntry", {'TotalPkts': int,'AvgPowerRcvd': float,'AvgInterferingPower': float})
LinkQualityNeighborEntry = TypedDict("LinkQualityNeighborEntry", {'RcvdPackets': LinkQualityRcvdPktsEntry, 'MCSIndex': int, 'TxPower': float})

class LinkQualityEntry(Protocol):
	@overload
	def __getitem__(self, key: Literal["LastRefreshedAt"]) -> float: ...
	@overload
	def __getitem__(self, key: Literal["InterferingPkts"]) -> InterferingPktsEntry: ...
	@overload
	def __getitem__(self, key: int) -> LinkQualityNeighborEntry: ...
	@overload
	def __setitem__(self, key: Literal["LastRefreshedAt"], value: float) -> None: ...
	@overload
	def __setitem__(self, key: Literal["InterferingPkts"], value: InterferingPktsEntry) -> None: ...
	@overload
	def __setitem__(self, key: int, value: LinkQualityNeighborEntry) -> None: ...
 
Link_Quality: dict[int, LinkQualityEntry] = defaultdict()

NodeStatsEntry = TypedDict('NodeStatsEntry', {'LastPST': float, 'PST': float, 'PktsServed': int, 'IL': Set[str], 'LastRefreshedAt': float})
Node_Stats: dict[int, NodeStatsEntry] = {}

Current_Topology: 'nx.Graph[int]' = nx.Graph()


Node_Locations_from_MobilityModel: NDArray[np.float64]
Node_NextWaypoints_from_MobilityModel: NDArray[np.float64]
time_rows: int
Node_Location_Waypoint: NDArray[np.float64]

DistToFurthestNodeInRange: list[float]



MAX_POWER_MULTIPLE = 2.75 # At 2.8 (with margin 1.0), node may overcome its 1 hop neighbor's transmission, which we don't want
MAXIMUM_ROUTING_RANGE = 1200
# ANDRES: Dictionary to save the nodes once they are created so that you can access the node with the node name (node "name" strings are the keys, and nodes (class NODE) are the values)
NODE_REGISTRY: Dict[str, 'Node'] = {}
# ANDRES: Dictionary to store the transmission powers in the corresponding index representing the 100m distance intervals.
DISTANCE_TRANSMISSION_POWER_DICT: Dict[int, Set[float]] = {}
# Reinsert the packet at source node upto 10 times.
NUMBERS_TO_WORDS: List[str] = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]

# Format of NAV entry-> Packet Id: Duration (= Curr time + Communication Duration) # in ns
NAV_Table: Dict[int, Dict[str, float]] = {}
Min_NAV_Expiry: list[float]


Interactive_Responses: dict[str, Any] = {}

@profile
def init_parameters(config_path: str, schema_path: str, only_load_configs: bool = False):
	load_config.load_config_stack(config_path, schema_path, target_module=sys.modules[__name__], interactive_override=Interactive_Responses)
	if not only_load_configs:
		print(f"Loaded config from {config_path}, using schema from {schema_path}")
		finish_setup()
		create_snapshot()


INCLUDE_DIRS = [
    "config",
    "mobility_bscap",
    "nodeTrajectories",
    "stubgen_output"
]

@profile
def create_snapshot():
	snapshot_dir = Path(RESULTS_FOLDER) / "snapshot"
	if snapshot_dir.exists() or snapshot_dir.with_suffix(".zip").resolve().exists():
		return
	snapshot_dir.mkdir(parents=True, exist_ok=True)

	PROJECT_ROOT = Path(".").resolve()

	for rel_dir in INCLUDE_DIRS:
		src_dir = PROJECT_ROOT / rel_dir
		if not src_dir.exists():
			print(f"Snapshot include dir not found: {src_dir}", file=sys.stderr)

		for path in src_dir.rglob("*"):
			if not path.is_file():
				continue
			
			rel_path = path.relative_to(PROJECT_ROOT)
			out = snapshot_dir / rel_path
			out.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(path, out)

	for path in PROJECT_ROOT.iterdir():
		if path.is_file():
			out = snapshot_dir / path.name
			shutil.copy2(path, out)
	
	metadata = {
		"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		"python_version": sys.version,
		"python_executable": sys.executable,
		"platform": sys.platform
	}

	with open(snapshot_dir / "metadata.json", "w") as f:
		json.dump(metadata, f, indent=4)

	try:
		with open(snapshot_dir / "requirements.txt", "w") as f:
			subprocess.run(
				["pip", "freeze"],
				stdout=f,
				stderr=subprocess.DEVNULL,
				check=True,
			)
	except Exception:
		pass
	
	print(f"Results folder: {RESULTS_FOLDER}", file=sys.stderr)

	with zipfile.ZipFile(
		snapshot_dir.with_suffix(".zip"),
		"w",
		compression=zipfile.ZIP_LZMA
	) as zip:
		for path in snapshot_dir.rglob("*"):
			if path.is_file():
				zip.write(path, arcname=path.relative_to(snapshot_dir))
    
	shutil.rmtree(snapshot_dir)

modified_stdout: bool = False

@profile
def finish_setup():
	### Storage location
	os.makedirs(RESULTS_FOLDER, exist_ok=True) # Stores logs and stats
	os.makedirs(NETWORK_GRAPHS, exist_ok=True)  # Stores topology graphs
	os.makedirs(PHEROMONE_GRAPHS, exist_ok=True)  # Stores topology graphs
	os.makedirs(TRAJECTORY_FOLDER, exist_ok=True) # Stores node trajectories
	os.makedirs(DISTANCE_TRANSMISSON_POWER_FOLDER, exist_ok=True)
	
	global modified_stdout
	if not modified_stdout:
		original_stdout = sys.stdout
		log_file = open(RESULTS_FOLDER+"run.out", "w")
		sys.stdout = log_file
		def close_log():
			log_file.flush()
			log_file.close()
			sys.stdout = original_stdout
		atexit.register(close_log)
		modified_stdout = True
	
	global Transmitting_Power, Maximum_Transmitting_Power
	Transmitting_Power = getTransmitPower(MAXIMUM_ROUTING_RANGE)
	Maximum_Transmitting_Power = getMaximumTransmitPower(DEFAULT_DATA_MCS_INDEX) # Refer to professor's notes
	print(f"{Transmitting_Power = }, {Maximum_Transmitting_Power = }")

	# NB: sim time are nanoseconds, distances are in meters, powers in watt

	# 802.11g parameters. 802.11g is the last standard not adopting MIMO. MIMO boost bitrate
	# using concurrent transmissions and other techniques. Values from wikipedia
	# https://en.wikipedia.org/wiki/IEEE_802.11
	# https://en.wikipedia.org/wiki/DCF_Interframe_Space
	# https://en.wikipedia.org/wiki/Short_Interframe_Space
	# Freq: 2.4 GHz, OFDM, 20 MHz bandwidth, 54 Mbit/s


	### Configuration flags

	# Sanity check
	if ((ENABLE_MCS_SNR) & (not CHANNEL_AWARE_ROUTE_COMPUTATION)):
		print("Enable CHANNEL_AWARE_ROUTE_COMPUTATION in parameters.py file in order to use Adaptive MCS-SNR!")
		exit()

	assert DEFAULT_DATA_MCS_INDEX >= CONTROL_MCS_INDEX, "MCS used for data packet should be greater than or equal to control packet"

	### SIMULATION PARAMETERS
	if (NODE_SPEED not in [20, 40, 50]):  # Sanity check . #TODO: Add samples for 50 m/s speed.
		print(f"No samples for node speed = {NODE_SPEED} m/s. Exit!")
		exit(1)

	assert PKT_GENERATION_START_TIME < PKT_GENERATION_END_TIME, "Packet generation start time must be less than packet gernation end time!"

	global Gateway_Node_ID
	Gateway_Node_ID = NUMBER_OF_NODES
	
	global MAX_Transmission_Range, MAX_Transmission_Range_Squared
	MAX_Transmission_Range = txRangeAtInterferenceLevel(Transmitting_Power, MCS_SNR_TABLE[CONTROL_MCS_INDEX]['MinSNR']) # SINR required to successfully decode a control (routing) packet
	MAX_Transmission_Range_Squared = MAX_Transmission_Range * MAX_Transmission_Range
	print(f"{MAX_Transmission_Range = }, {MAX_Transmission_Range_Squared = }")
	
	global DistToFurthestNodeInRange
	DistToFurthestNodeInRange = [0] * NUMBER_OF_NODES
	
	global INTF_Registry
	# AADITYA: List to store nodes' interference
	INTF_Registry = [0] * (NUMBER_OF_NODES + NUMBER_OF_BS)
	
	global Routes
	# Route Assumptions: Unique source nodes, node index starts with 0
	if MOBILE_SCENARIO: # ROUTES = [[0,0],...]   # SHREY*: default [src,dst]
		Routes = [] # SD: Src node will be selected after target discovery
	elif FORCED_ROUTES:
		Routes = [(route, (-1, -1)) for route in FORCED_ROUTES]
	else:
		Routes = []		

	initializeLinkQuality(0.0)  # in seconds
	initializeNodeStats(0.0)
	
	# Initialize NAV Table
	global NAV_Table, Min_NAV_Expiry
	NAV_Table = {}
	for eachNode in range(0, NUMBER_OF_NODES):
		NAV_Table[eachNode] = {}
	NAV_Table[Gateway_Node_ID] = {}

	Min_NAV_Expiry = [float("inf")] * (NUMBER_OF_NODES+1)

	global Node_Locations_from_MobilityModel, Node_NextWaypoints_from_MobilityModel, time_rows, Node_Location_Waypoint
	#SHREY*
	Node_Locations_from_MobilityModel = np.zeros((NUMBER_OF_NODES, 2))
	Node_NextWaypoints_from_MobilityModel = np.zeros((NUMBER_OF_NODES, 2))

	# TRAJECTORY_FOLDER 
	time_rows = int(SIM_TIME/1e9) - int(PKT_GENERATION_START_SEC)
	Node_Location_Waypoint = np.zeros( (NUMBER_OF_NODES, time_rows , 4) )

	print("parameters.py INITIALIZED")
	
	# NODE_LOCATIONS_MOBILITY = np.zeros(parameters.NUMBER_OF_NODES, 2)
	# UAV_ENERGIES = np.ones(parameters. NUMBER_OF_NODES, 2) * np.float("inf")

	#extra
	bins=11
	MOBILITY.connectivity_histogram = np.zeros(bins)

	# print(f"Dropped nodes: {num_failures}, Active nodes: {num_active} (Between 1000s and 3000s)")

	# Energy distribution
	active_nodes_energy: NDArray[np.float64] = np.full(MOBILITY.NUM_ACTIVE, MOBILITY.MAX_INITIAL_ENERGY, dtype=np.float64)  # Fully charged UAVs
	failed_nodes_energy: NDArray[np.float64] = np.random.uniform(100, 300, MOBILITY.NUM_FAILURES)  # Failing UAVs have lower energy

	# Combine and shuffle energy values
	MOBILITY.NODE_ENERGY = np.concatenate((active_nodes_energy, failed_nodes_energy)).astype(np.float64)
	np.random.shuffle(MOBILITY.NODE_ENERGY)

	# Add base station energy
	MOBILITY.NODE_ENERGY = np.append(MOBILITY.NODE_ENERGY, MOBILITY.MAX_INITIAL_ENERGY)

	# #---------------------------------------------------------------------------------------------------------------------
	# # UAV Energies at t=0
	# stats.UAV_energies[0,:] = node_Energy
	# #---------------------------------------------------------------
	
	global HIGH_PRIORITY, LOW_PRIORITY
	HIGH_PRIORITY = 2
	LOW_PRIORITY = 1

	print("mobility parameters INITIALIZED")

@profile
@lru_cache(1024)
def getTransmitPower(distance: float, margin: float = 1.05, mcs_index: Optional[int] = None, noise: Optional[float] = None,
                     		interference: Optional[float] = None, wavelength: Optional[float] = None, antenna_gain: Optional[float] = None) -> float:
    
	if mcs_index is None: mcs_index = CONTROL_MCS_INDEX
	if noise is None: noise = NOISE_FLOOR
	if interference is None: interference = 0
	if wavelength is None: wavelength = WAVELENGTH
	if antenna_gain is None: antenna_gain = 1
	
	snr = MCS_SNR_TABLE[mcs_index]['MinSNR']
	
	P_r = snr * (noise+interference)
	attenuation = (wavelength / (4 * pi * distance)) ** 2
	
	P_t = P_r / (attenuation * antenna_gain)
	
	# if PRINT_LOGS:
	#   print(f"{snr = }, total noise = {noise+interference}, {P_r = }, {attenuation = }, {P_t = }")
	
	return P_t * margin

@profile
@lru_cache(16)
def getMaximumTransmitPower(MCS_index: int) -> float:
	return getTransmitPower(distance=MAXIMUM_ROUTING_RANGE, mcs_index=MCS_index, margin=1) * MAX_POWER_MULTIPLE

@profile
def set_bs_ids(bs_ids: list[int]) -> None:
	global BSIDs
	BSIDs = bs_ids.copy()


# Sh: Get transmission rate for the given MCS index
@profile
def getBitTxRate(MCS_index: int) -> float:
	return (1/(MCS_SNR_TABLE[MCS_index]['DataRate'] * 1e6)) * 1e9 # Before: (1000/MCS_SNR_TABLE[MCS_index]['DataRate'])


# Sh: Compute max transmission range assuming no external interference, most robust MCS index, max allowed tx power and min rx sensitivity. Used for routing.
@profile
def txRangeAtInterferenceLevel(tx_power: float, req_sinr: Optional[float] = None, interference_power: float = 0) -> float:
	if req_sinr is None: req_sinr = MCS_SNR_TABLE[CONTROL_MCS_INDEX]['MinSNR']
	return math.sqrt(tx_power/max(RADIO_SENSITIVITY, req_sinr*(interference_power+NOISE_FLOOR)))*(WAVELENGTH/(4*pi))

# Sh: CTS Timeout changes based on MCS index
# 2/12/26 Removed external_interference parameter
@profile
def computeCtsTimeout(tx_power: float) -> float:
	# SD: All MAC control packets (RTS, CTS) are to sent at MCS0 to be able to decoded at max transmit distance.
	# bit_tx_rate = getBitTxRate(MCS_index) # SD: commented
	# max_tx_range = txRangeAtInterferenceLevel(tx_power, MCS_SNR_TABLE[MCS_index]['MinSNR'], external_interference) # SD:
	# cts timeout = Transmission time for RTS and CTS pakcets + rtt (or 2*propagation delay) + sifs
	# cts_timeout = (RTS_LENGTH + CTS_LENGTH + 2*PHY_HEADER_LENGTH) * bit_tx_rate + 2 * round((max_tx_range / c) * pow(10,9), 0) + SIFS_DURATION
	#print("CTS timeout: %s" % (cts_timeout))
	MCS_index = CONTROL_MCS_INDEX

	default_bit_tx_rate= getBitTxRate(MCS_index) # SD: generally MCS0
	max_tx_range = txRangeAtInterferenceLevel(tx_power, MCS_SNR_TABLE[MCS_index]['MinSNR']) # SD: This is max distance the RTS can be decoded when default tx power and MCS0 is used. IMP_NOTE: Do not use MAX_TRANSMISSION_RANGE (global value), if P_reduced < P_default.
	cts_timeout = (RTS_LENGTH + CTS_LENGTH + 2*PHY_HEADER_LENGTH) * default_bit_tx_rate + 2 * round((max_tx_range / c) * pow(10,9), 0) + SIFS_DURATION

	return cts_timeout


# Sh: ACK Timeout value changes based on MCS index
# ANDRES: added txPower parameter so that it can change with the users sending power
# 2/12/26 Removed external_interference parameter
@profile
def computeAckTimeout(tx_power_rts: float, tx_power_data: float, number_of_packets: int = 1) -> float:
	Data_MCS_index = DEFAULT_DATA_MCS_INDEX
	Control_MCS_index = CONTROL_MCS_INDEX

	control_bit_tx_rate = getBitTxRate(Control_MCS_index) # SD: All MAC control packets (RTS,CTS, ACK) are to sent at parameters.CONTROL_MCS_INDEX
	
	coding_rate = 1#MCS_SNR_TABLE[Data_MCS_index]['Coding']

	data_bit_tx_rate = getBitTxRate(Data_MCS_index) # SD: For DATA packet uses bit_tx_rate based on the given MCS index
	max_tx_range = txRangeAtInterferenceLevel(tx_power_data, MCS_SNR_TABLE[Data_MCS_index]['MinSNR']) # SD:
	# max_tx_range = MAX_TRANSMISSION_RANGE # SD: IMP_NOTE: Do not use MAX_TRANSMISSION_RANGE (global value), if P_reduced < P_default.
	# SD: ACK uses MCS0 (default).
	
	data_time = ((number_of_packets*MAX_MAC_PAYLOAD_LENGTH) + MAC_HEADER_LENGTH + PHY_HEADER_LENGTH) * data_bit_tx_rate / coding_rate
	ack_timeout = data_time + 2 * round((max_tx_range / c) * pow(10,9), 0) + SIFS_DURATION + (ACK_LENGTH + PHY_HEADER_LENGTH) * control_bit_tx_rate
	ack_timeout += computeCtsTimeout(tx_power_rts) + SIFS_DURATION

	return ack_timeout

@profile
def initializeROUTE_DETAILS(ROUTES: list[tuple[list[int], tuple[float, float]]]) -> None:
	for eachRoute, target in ROUTES:
		routeIdx = len(Route_Details)
		Route_Details[routeIdx] = {
      		'Route': eachRoute, 
			'MCS_Index': [DEFAULT_DATA_MCS_INDEX for _ in range(len(eachRoute)-1)], 
			'Priority': 1+routeIdx if USE_FLOW_PRIORITY else 1,
			'ValidRoute': False, 
			'HQRoute': False,
			'RouteNodeDensity':[None for _ in range(len(eachRoute))],
			'InterferingLinks': 0,
			'Target': target,
		}
	print("Route initialization: ", Route_Details)
# initializeROUTE_DETAILS(ROUTES) 


#def getOneHopNghbrs(nodeId):
#  return Current_Topology.neighbors(nodeId)

# Maintain link quality related stats for each link in the network
@profile
def initializeLinkQuality(currTime: float) -> None:
	for eachNode in range(0, NUMBER_OF_NODES+1): # +1 for Gateway node
		Link_Quality[eachNode] = cast(LinkQualityEntry, {})
		Link_Quality[eachNode]['LastRefreshedAt'] = currTime  # in Seconds,
		Link_Quality[eachNode]['InterferingPkts'] = InterferingPktsEntry({'TotalPkts':0, 'AvgPower':0.0}) # Initial estimate for links which did not receive packet in the last interval
		# Create the following for each reachable neighbor
		'''for eachNghbr in range(0,NUMBER_OF_NODES):
			LINK_QUALITY[eachNode][eachNghbr] = {}
			LINK_QUALITY[eachNode][eachNghbr]['RcvdPackets'] = {'TotalPkts':0,'AvgPowerRcvd':0.0,'AvgInterferingPower':0.0} # Power in watts
			LINK_QUALITY[eachNode][eachNghbr]['MCSIndex'] = 0
			LINK_QUALITY[eachNode][eachNghbr]['TxPower'] = TRANSMITTING_POWER  # in watts'''
#print("parameters.py: Link quality initialization",LINK_QUALITY)

# Maintain node statistics, such as PST (packet service time) and IL (number of interfering links)
@profile
def initializeNodeStats(currTime: float) -> None:
	for eachNode in range(0, NUMBER_OF_NODES+1): # +1 for Gateway node
		if (currTime <= 0.0): # First time creation
			Node_Stats[eachNode] = {'LastPST': 0, # Used in computing moving average of node's PST value
									'PST': 0,
									'PktsServed': 0,
									'IL': set(),
									'LastRefreshedAt': currTime }  # in Seconds
		else:
			eachNodeStats = Node_Stats[eachNode]
			eachNodeStats['LastPST'] = eachNodeStats['PST']
			eachNodeStats['PST'] = 0
			eachNodeStats['PktsServed'] = 0
			eachNodeStats['IL'] = set()
			eachNodeStats['LastRefreshedAt'] = currTime  # in Seconds
#print("parameters.py: Node statistics initialization",NODE_STATS)


# Delete outdated NAV entries at a given node
@profile
def deleteOldNAVEntriesAtNode(nodeId: int, cutoffTime: float) -> None:
	nodeEntries = NAV_Table[nodeId]

	if not nodeEntries:
		Min_NAV_Expiry[nodeId] = float('inf')
		return

	new_min = float('inf')
	#print(f"At node:{nodeId}, cutofftime: {cutoffTime}, Delete from NAV entries: {nodeEntries}")
	for pktID, duration in list(nodeEntries.items()):
		if duration < cutoffTime:
			del nodeEntries[pktID]
		else:
			if duration < new_min:
				new_min = duration

	if nodeEntries:
		Min_NAV_Expiry[nodeId] = new_min
	else:
		Min_NAV_Expiry[nodeId] = float('inf')
	#print(NAV_Table)
	return

# Delete outdated NAV entries at all nodes
@profile
def deleteAllOldNAVEntries(cutoffTime: float) -> None:
	for eachNode in NAV_Table:
		deleteOldNAVEntriesAtNode(eachNode, cutoffTime)
	#print(NAV_Table)
	return

# Get current NAV at a node
@profile
def getMaxNAV(nodeId: int, cutoffTime: float) -> float:
    nodeEntries = NAV_Table[nodeId]

    if not nodeEntries:
        return cutoffTime - 1

    max_val = cutoffTime - 1
    min_expiry = Min_NAV_Expiry[nodeId]

    # Fast path: no expiration possible
    if cutoffTime <= min_expiry:
        for duration in nodeEntries.values():
            if duration > max_val:
                max_val = duration
        return max_val

    # Slow path: expiration possible
    new_min = float('inf')
    for pktID, duration in list(nodeEntries.items()):
        if duration < cutoffTime:
            del nodeEntries[pktID]
        else:
            if duration > max_val:
                max_val = duration
            if duration < new_min:
                new_min = duration

    if nodeEntries:
        Min_NAV_Expiry[nodeId] = new_min
    else:
        Min_NAV_Expiry[nodeId] = float('inf')

    return max_val

# Deleting a NAV entry from all nodes
@profile
def delNAVEntry(packetID: str, nodeId: Optional[int] = None) -> None:
	if nodeId is None:
		for eachNode, nodeEntries in NAV_Table.items():
			if packetID in nodeEntries:
				removed_expiry = nodeEntries.pop(packetID)

				# Recompute min only if necessary
				if removed_expiry == Min_NAV_Expiry[eachNode]:
					if nodeEntries:
						Min_NAV_Expiry[eachNode] = min(nodeEntries.values())
					else:
						Min_NAV_Expiry[eachNode] = float('inf')
	else:
		nodeEntries = NAV_Table[nodeId]
		if packetID in nodeEntries:
			removed_expiry = nodeEntries.pop(packetID)

			if removed_expiry == Min_NAV_Expiry[nodeId]:
				if nodeEntries:
					Min_NAV_Expiry[nodeId] = min(nodeEntries.values())
				else:
					Min_NAV_Expiry[nodeId] = float('inf')
	return


# SD: Added function to initialize NODE QUEUES and FLOW PACKET GENERATION
@profile
def initialize_NODES_FLOWLOGS_QUEUE_PACKETGENERATION(env: simpy.Environment, nodes: list['Node']) -> None:
	import stats
	statistics = stats.Stats()
	import csv
#---------------------------------------------------------------------------------------------------------------------------------------------
	# print('P1', len(ROUTES))
	# MC: Remove expired packets from queue, Sh: Queue management (reshuffle queue based on survivability score and discard packets that are soon to expire)
	if PACKET_TTL <= SIM_TIME:
		for eachNodeIdx in range(len(nodes)): 
			env.process(nodes[eachNodeIdx].DropPacketDueToExpiry(nodes[eachNodeIdx].q))
			# Sh: Queue management
			if ENABLE_Q_MNGMNT:
				env.process(nodes[eachNodeIdx].QueueManagement(nodes[eachNodeIdx].q))
	
	# # Sh: Create interrupt to periodically update node locaiton, and thereby, network topology and routes
	# env.process(rf.updateNetworkTopology(env,nodes))  #SHREY* commented , now called below with yield

	src_active: set[int] = set()
	# Sh: Setup source-destinaiton pairs
	for eachRouteIdx in Route_Details:
		eachRoute = Route_Details[eachRouteIdx]['Route']

		# Sanity check condition. Discard route if (a) single node in a route, (b) src == dst
		if ((len(eachRoute) < 2) or (eachRoute[0] == eachRoute[-1])):
			continue
		src = eachRoute[0]; nexthop = eachRoute[1]; dst = eachRoute[-1]
		assert src not in src_active, "Multiple flows can not start at the same source. If changing, look into packet ids and compiling results in mobile cases."
		src_active.add(src)
		# Create stat log file for this flow
		flowID = statistics.createFlowID(nodes[src].name, nodes[dst].name)
		with open(RESULTS_FOLDER+flowID, 'w', newline='') as f_:
			writer = csv.writer(f_)
			writer.writerow(["Pkt ID","Route","Generation Time (in s)", "First Tx At (in s)","Delay (in s)","Times Packet Forwarded","Retransmissions","Packet Size (in bytes)",
							"Tx Power (in watt)","Min. SINR Along Route", "SINR at Flow Dest","Min. Data Rate Along Route (in Mbps)","Data Rate at Flow Src (in Mbps)"])
			f_.close()
		
		if PRINT_LOGS:
				print("\npheromone.py: Set up packet generation for route: %s, src: %s, nextHop: %s, dst: %s\n" % (eachRoute, src, nexthop, dst))
		#env.process(nodes[src].keepSending(TARGET_PKT_GENERATION_RATE, nodes[nexthop].name, nodes[src].name, nodes[dst].name, CONTROL_MCS_INDEX)) # Sh
		
		# MC: Setup periodic packet generation for this source
		env.process(nodes[src].PacketGeneration(eachRouteIdx))
		# print('P2', src,dst)
	env.process(phy.cleanup_slot_information(env))
	env.process(phy.update_intf_table(env))
	for name in NODE_REGISTRY:
		env.process(NODE_REGISTRY[name].detect_congestion())