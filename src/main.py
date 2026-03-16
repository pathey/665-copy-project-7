'''
This code has been adapted from https://github.com/riccardomicheletto/SPEproject-802.11DCFsim
Contact: Prof. Sunil Kumar, SDSU, California, USA
Last modified on: Nov 11, 2024

Key Features: 
* Select the best modulation and coding scheme (MCS) index based on interference and noise level at the intended receiver. 
* Introduced new functionality to log stats for each flow.
* Has packet retransmission capabilities.
* Integrated route selecton metrics (hop count, link cost and MCA). Code has been tested for static and mobile scenarios with single and/or multi-flows.
* Introduced queue to model flow priority and packet drop due to expiry and/or buffer overflow.
* Added network allocation vector (NAV) functionality and support to emulate RTS/CTS prior to data packet transmission.
* Code optimized to reduce computational time. E.g., packet reception events are generated for nodes in range (see ether.py), and use of STEP_SIZE (see parameters.py).
* Functionality to reinsert packets at the source node which were dropped at the intermediate node(s) due to sudden route change. See ENABLE_PACKET_REINSERTION flag in parameters.py.
* Automated running multiple simulations (see GenerateResults.sh script), and result extraction (see CompileResult.py file). 
To run single simulation, execute command 'python3 main.py'. For multiple runs, execute command './GenerateResults.sh'.

Set ENABLE_MCS_SNR to True in parameters.py file to use MCS-SNR feature.
'''
from graphlib import CycleError, TopologicalSorter
from operator import attrgetter
import os
import sys
from pathlib import Path

import yaml

compiled_packages = Path("compiled_packages").resolve().as_posix()

sys.path.insert(0, compiled_packages)
if hasattr(os, 'add_dll_directory'):
	os.add_dll_directory(compiled_packages)
else:
	os.environ['PATH'] = '/path/to/dll_or_so' + os.pathsep + os.environ['PATH']

import argparse
import ast
import copy
from dataclasses import dataclass, fields, is_dataclass
import pickle
from typing import TYPE_CHECKING, Any, Callable, Generator, Mapping, ParamSpec, TypeVar, cast
from collections.abc import Mapping as MappingABC

from numpy.typing import NDArray

from simpy.events import Timeout
# import psutil

# process = psutil.Process()

import matplotlib
import matplotlib.style as mplstyle
mplstyle.use('fast')
matplotlib.use('agg')


import simpy
import random
import numpy as np

import node
import ether
import parameters
import stats

# Sh
import RouteFunctions as rf
import csv

#mobility.py
# from mobility_bscap import mobility
from mobility_bscap import pheromone as ph

from prompt_toolkit import prompt

T = TypeVar("T")


from line_profiler import profile # type: ignore[reportAssignmentType]

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

@profile
def main():
	env = simpy.Environment()
	eth = ether.Ether(env)
	statistics = stats.Stats()

	nodes: list[Node] = []

	if parameters.MOBILE_SCENARIO:
		parameters.finish_setup()
		
		for i in range(0, parameters.NUMBER_OF_NODES+1): # +1 for Gateway node
			name = "Node" + str(i)
			lat,lon =  0., 0.  #rf.get_NodeLocation_from_MobilityModel( i ) #SHREY* initial to zero; will be assigned after starting mobility ann updateNetworkTopology processes 
			nodes.append(node.Node(env, name, eth, lat, lon, statistics, parameters.Transmitting_Power, external_interference=parameters.EXT_INTERFERENCE.get(i, 0)))
			
			# Create log file to record queue info of this node
			with open(parameters.RESULTS_FOLDER+name+".csv", 'w', newline='') as f_:
				writer = csv.writer(f_)
				writer.writerow(stats.QCOLS)
	else:
		positions: list[tuple[float, float]] = load_topology(Path(parameters.STATIC_TOPOLOGY_PATH))
		positions.append((0, 0)) # Dummy for Gateway Node

		parameters.NUMBER_OF_NODES = len(positions)-1
		parameters.Gateway_Node_ID = parameters.NUMBER_OF_NODES # Find if there way to not have to setup parameters again
		parameters.MOBILITY.NUM_FAILURES = int((parameters.NUMBER_OF_NODES - 1) * (parameters.MOBILITY.NODE_FAILURE_PERCENTAGE / 100))
		parameters.MOBILITY.NUM_ACTIVE = (parameters.NUMBER_OF_NODES - 1) - parameters.MOBILITY.NUM_FAILURES
		parameters.MOBILITY.ENERGY_DEPLETION_RATE = np.full(parameters.NUMBER_OF_NODES - parameters.NUMBER_OF_BS, parameters.MOBILITY.MAX_ENERGY_DEPLETION_RATE)

		parameters.finish_setup()
		
		for i, position in enumerate(positions):
			name = "Node" + str(i)
			
			lat,lon = position
			nodes.append(node.Node(env, name, eth, lat, lon, statistics, parameters.Transmitting_Power, external_interference=parameters.EXT_INTERFERENCE.get(i, 0)))
			
			# Create log file to record queue info of this node
			with open(parameters.RESULTS_FOLDER+name+".csv", 'w', newline='') as f_:
				writer = csv.writer(f_)
				writer.writerow(stats.QCOLS)

		# eth.registerNodes(nodes) # AT: All nodes should have been created by this point


# BSCAP mobility model-------------------------------------------------------------------------------------------------------------------------------------

	# nAgents = parameters.NUMBER_OF_NODES # no. of nodes+1BS
	BETA_TYPE= 1.5 # tuning parameter value fixed to beta= 1.5
	STATS_INTERVAL = 3 # STATS_INTERVAL=100
	SIM_TIME_MOBILITY = int(parameters.SIM_TIME / 1e9) # total simulation time (endtime) converting from ns to seconds
	print("SIM_TIME_MOBILITY = ", SIM_TIME_MOBILITY)
	Pipe_neighbor_density_threshold = 2 # 1-hop neighbor density threshold for nodes in active route 
	# Setting same max TX RANGE to all Nodes for mobility purposes
	node_max_tx_ranges = parameters.txRangeAtInterferenceLevel(parameters.Transmitting_Power, 
																		parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'])
	print("MAX TRANSMISSION RANGE for all nodes - mobility.py = ", node_max_tx_ranges)
	print("DATA_RATE = ", parameters.PAYLOAD_DATA_RATE, " No._Nodes = ", parameters.NUMBER_OF_NODES," Map_Size = ", parameters.AREA_X ,"x", parameters.AREA_X )


	swarm = ph.uav_swarm(env, evaporation_rate = 0.006, diffusion_rate = 0.006, use_connect = True, #init_posX=init_posX, init_posY=init_posY,
						nAgents = parameters.NUMBER_OF_NODES-parameters.NUMBER_OF_BS, nBaseStations = parameters.NUMBER_OF_BS, map_size = parameters.AREA_X,
						connectivity_scheme = 1, collision_avoidance = True, collision_buffer = float(200),
						stats_interval = STATS_INTERVAL,fwd_scheme = 5, hop_dist = 2,
						map_resolution = 100, transmission_range = node_max_tx_ranges, alpha_type = BETA_TYPE,
						target_location = parameters.MOBILITY.TARGET_LOCATION, Pipe_neighbor_density_threshold = Pipe_neighbor_density_threshold,
						drop_percent = parameters.MOBILITY.NODE_FAILURE_PERCENTAGE)
	
	parameters.set_bs_ids(swarm.bs_iDs)

	#SHREY*
	ith_episode=0 #NOT USED
	_ = env.process(swarm.sim_start_3d(nodes, SIM_TIME_MOBILITY, ith_episode, drawUAVs=False, drawMap=True, plotInterval=10)) #SHREY* starting moblity model process
	# print(res)
	print("MOBILITY PROCESS (^) and NETWORK UPDATE (rf.updateNetworkTopology(env,nodes) HAS BEEN STARTED ")

# #---------------------------------------------------------------------------------------------------------------------------------------------
#     # print("AAT main.py t=0 ", parameters.Node_Locations_from_MobilityModel)

#     # MC: Remove expired packets from queue, Sh: Queue management (reshuffle queue based on survivability score and discard packets that are soon to expire)
#     if parameters.PACKET_TTL <= parameters.SIM_TIME:
#         for eachNodeIdx in range(len(nodes)): 
#             env.process(nodes[eachNodeIdx].DropPacketDueToExpiry(nodes[eachNodeIdx].q))
#             # Sh: Queue management
#             if parameters.ENABLE_Q_MNGMNT:
#                 env.process(nodes[eachNodeIdx].QueueManagement(nodes[eachNodeIdx].q))
	
#     # # Sh: Create interrupt to periodically update node locaiton, and thereby, network topology and routes
#     # env.process(rf.updateNetworkTopology(env,nodes))  #SHREY* commented , now called in pheromone.py through mobility.start_mobility(env, nodes)

#     # Sh: Setup source-destinaiton pairs
#     for eachRouteIdx in parameters.Route_Details:
#         eachRoute = parameters.Route_Details[eachRouteIdx]['Route']

#         # Sanity check condition. Discard route if (a) single node in a route, (b) src == dst
#         if ((len(eachRoute) < 2) or (eachRoute[0] == eachRoute[-1])):
#             continue
#         src = eachRoute[0]; nexthop = eachRoute[1]; dst = eachRoute[-1]
		
		# # Create stat log file for this flow
		# flowID = statistics.createFlowID(nodes[src].name,nodes[dst].name)
		# with open(parameters.RESULTS_FOLDER+flowID, 'w', newline='') as f_:
		#     writer = csv.writer(f_)
		#     writer.writerow(stats.FCOLS)
		
#         if parameters.PRINT_LOGS:
#             print("\nmain.py: Set up packet generation for route: %s, src: %s, nextHop: %s, dst: %s\n" % (eachRoute, src, nexthop, dst))
#         #env.process(nodes[src].keepSending(parameters.TARGET_PKT_GENERATION_RATE, nodes[nexthop].name, nodes[src].name, nodes[dst].name, parameters.CONTROL_MCS_INDEX)) # Sh
		
#         # MC: Setup periodic packet generation for this source
#         env.process(nodes[src].PacketGeneration(nodes[src].name, nodes[dst].name))
		
	if not parameters.PRINT_LOGS:
		env.process(printProgress(env))


	
	# print("Before run: ", process.memory_info().rss / 1024 ** 2, "MB rss, ", process.memory_info().vms / 1024 ** 2, "MB vms")
	env.run(until=parameters.SIM_TIME)
	# print("After run: ", process.memory_info().rss / 1024 ** 2, "MB rss, ", process.memory_info().vms / 1024 ** 2, "MB vms")
	
	@profile
	def save_object(obj: Any, filename: str) -> None:
		with open(filename, 'wb') as output:  # Overwrites any existing file.
			pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	
	if swarm.stats:
		savestats = SaveStats()
		savestats.r_coverage = copy.deepcopy(swarm.stats.coverage)
		savestats.r_no_connected_comp = copy.deepcopy(swarm.stats.no_connected_comp)
		savestats.r_avg_deg_conn = copy.deepcopy(swarm.stats.avg_deg_conn)
		savestats.r_largest_subgraph = copy.deepcopy(swarm.stats.largest_subgraph)
		savestats.r_frequency = copy.deepcopy(swarm.stats.frequency)
		savestats.r_is_biconnected_Gaint_Comp = copy.deepcopy(swarm.stats.is_biconnected_Gaint_Comp)
		savestats.r_total_time_connected_to_BS = copy.deepcopy(swarm.stats.total_time_connected_to_BS)
		
		savestats.coverage = copy.deepcopy(swarm.stats.coverage_after_genstart)
		savestats.no_connected_comp = copy.deepcopy(swarm.stats.no_connected_comp_after_genstart)
		savestats.avg_deg_conn = copy.deepcopy(swarm.stats.avg_deg_conn_after_genstart)
		savestats.largest_subgraph = copy.deepcopy(swarm.stats.largest_subgraph_after_genstart)
		savestats.is_biconnected_Gaint_Comp = copy.deepcopy(swarm.stats.is_biconnected_Gaint_Comp_after_genstart)
		savestats.frequency = copy.deepcopy(swarm.stats.frequency_after_genstart)
		savestats.total_time_connected_to_BS = copy.deepcopy(swarm.stats.total_time_connected_to_BS_after_genstart)
		savestats.runtime = copy.deepcopy(swarm.stats.runtime)

		savestats.g_fairness_after_genstart = copy.deepcopy(swarm.stats.fairness_after_genstart)
		savestats.g_freq_subgraph_sizes_after_genstart = copy.deepcopy(swarm.stats.freq_subgraph_sizes_after_genstart)    

		save_object(savestats, parameters.RESULTS_FOLDER + "mobility_stats.pkl")

	# Sh: Create gif to visualize changes in network topology
	rf.createGif()

	# ANDRES: create the gif for the distance-transmission power graph if the adaptive power scheme is enabled
	# if parameters.ENABLE_POWER_CONTROL:
	rf.createGif_Distance_TxPower()

	statistics.plotCumulativePackets()
	statistics.plotThroughput()
	statistics.plotThroughputMs()
	statistics.plotDelays()
	statistics.plotRetransmissions()
	
	for each_node in parameters.NODE_REGISTRY.values():
		print(f"{each_node.name}")
		print(f"Avg Received Power: {each_node.mac.phy.total_power_received.mean():.2e}, ", end='')
		for phy_name, reception_stats in each_node.mac.phy.power_from_others.items():
			print(f"{phy_name}: [Avg:{reception_stats.mean():.2e}, Max:{reception_stats.max:.2e}, Min:{reception_stats.min:.2e}], ", end='')
		print('')
		print(f"Avg INTF: {each_node.mac.phy.total_intf_received.mean():.2e}, ", end='')
		for phy_name, intf_stats in each_node.mac.phy.intf_from_others.items():
			print(f"{phy_name}: [Avg:{intf_stats.mean():.2e}, Max:{intf_stats.max:.2e}, Min:{intf_stats.min:.2e}], ", end='')
		print('')

		print(f"Self Tx power: RTS: (Avg: {each_node.mac.phy.tracker_power_rts.mean():.4f}, Min: {each_node.mac.phy.tracker_power_rts.min:.4f}, Max: {each_node.mac.phy.tracker_power_rts.max:.4f}), CTS: (Avg: {each_node.mac.phy.tracker_power_cts.mean():.4f}, Min: {each_node.mac.phy.tracker_power_cts.min:.4f}, Max: {each_node.mac.phy.tracker_power_cts.max:.4f}), ACK: (Avg: {each_node.mac.phy.tracker_power_ack.mean():.4f}, Min: {each_node.mac.phy.tracker_power_ack.min:.4f}, Max: {each_node.mac.phy.tracker_power_ack.max:.4f}), Data: (Avg: {each_node.mac.phy.tracker_power_data.mean():.4f}, Min: {each_node.mac.phy.tracker_power_data.min:.4f}, Max: {each_node.mac.phy.tracker_power_data.max:.4f})")
		print(f"CTS retry: {each_node.mac.cts_retry}, ACK retry: {each_node.mac.ack_retry}, # Receiver Drops: {each_node.mac.phy.receiver_drops}, # Pkts Receiver Drops: {each_node.mac.phy.receiver_dropped_packet}, # Sent ACKs: {each_node.mac.sent_ack}, # Pkts ACK sent for: {each_node.mac.send_ack_per_packet}, # Retransmissions: {each_node.mac.num_retransmissions}, # Pkts Retransmitted: {each_node.mac.num_pkts_retransmitted}, # Receptions: {each_node.mac.num_received}, # Pkts Received: {each_node.mac.num_packets_received}, # Forwards: {each_node.mac.num_forwards}, # Pkts Forwarded: {each_node.mac.num_packets_forwarded}\n")
	
	timeout_cols = [
		"Node Name",
		"CTS Retry",
		"ACK Retry",
		"# Receiver Drops",
		"# Pkts Receiver Drops",
		"# Sent ACKs",
		"# Pkts ACK Sent For",
		"# Retransmissions",
		"# Pkts Retransmitted",
		"# Receptions",
		"# Pkts Received",
		"# Forwards",
		"# Pkts Forwarded"
	]

	with open(Path(parameters.RESULTS_FOLDER) / "timeout_stats.csv", mode="w", newline="") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=timeout_cols)
		writer.writeheader()

		for each_node in parameters.NODE_REGISTRY.values():
			row: dict[str, Any] = {
				"Node Name": each_node.name,
				"CTS Retry": each_node.mac.cts_retry,
				"ACK Retry": each_node.mac.ack_retry,
				"# Receiver Drops": each_node.mac.phy.receiver_drops,
				"# Pkts Receiver Drops": each_node.mac.phy.receiver_dropped_packet,
				"# Sent ACKs": each_node.mac.sent_ack,
				"# Pkts ACK Sent For": each_node.mac.send_ack_per_packet,
				"# Retransmissions": each_node.mac.num_retransmissions,
				"# Pkts Retransmitted": each_node.mac.num_pkts_retransmitted,
				"# Receptions": each_node.mac.num_received,
				"# Pkts Received": each_node.mac.num_packets_received,
				"# Forwards": each_node.mac.num_forwards,
				"# Pkts Forwarded": each_node.mac.num_packets_forwarded
			}
			writer.writerow(row)


def _parse_coord(token: str, prev: float) -> float:
	token = token.strip()
	if token.startswith("(") and token.endswith(")"):
		token = token[1:-1]
		prev = 0
	if token.startswith("+"):
		return prev + float(token[1:])
	if token.startswith("-"):
		return prev - float(token[1:])
	return float(token)

def load_static_topology(path: Path) -> list[tuple[float, float]]:
	positions: list[tuple[float, float]] = []

	prev_x = prev_y = 0.0

	with open(path) as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"): continue

			x_tok, y_tok = line.split(",", 1)

			x = _parse_coord(x_tok, prev_x)
			y = _parse_coord(y_tok, prev_y)

			positions.append((x, y))
			prev_x, prev_y = x, y

	return positions

def load_time_topology(path: Path, tick: int) -> list[tuple[float, float]]:
	positions: list[tuple[float, float]] = []

	for i in range(parameters.NUMBER_OF_NODES + 1):
		x, y = rf.nodeLocation(f"Node{i}", tick)
		positions.append((x, y))

	return positions

def load_topology(path: Path) -> list[tuple[float, float]]:

	if path.is_dir():
		return load_time_topology(path, 0)

	if path.is_file():
		return load_static_topology(path)

	raise ValueError(f"Topology path does not exist: {path}")


class SaveStats(object):
	def __init__(self) -> None:
		self.coverage: list[float] = []
		self.no_connected_comp: list[int] = []
		self.avg_deg_conn: list[np.floating] = []
		self.largest_subgraph: list[float] = []
		self.is_biconnected_Gaint_Comp: list[bool] = []
		self.frequency: NDArray[np.int32] = np.array(0, dtype=np.int32)
		self.total_time_connected_to_BS: NDArray[np.int8] = np.array(0, dtype=np.int8)
		self.runtime: list[float] = []
		
		self.r_coverage: list[float] = []
		self.r_no_connected_comp: list[int] = []
		self.r_avg_deg_conn: list[np.floating] = []
		self.r_largest_subgraph: list[float] = []
		self.r_is_biconnected_Gaint_Comp: list[bool] = []
		self.r_frequency: NDArray[np.int32] = np.array(0, dtype=np.int32)
		self.r_total_time_connected_to_BS: NDArray[np.int8] = np.array(0, dtype=np.int8)
		
		self.g_fairness_after_genstart: list[float | np.floating] = []
		self.g_freq_subgraph_sizes_after_genstart: NDArray[np.floating] = np.array(0, dtype=np.float64)

@profile
def printProgress(env: simpy.Environment) -> Generator[Timeout, None, None]:
	while True:
		print('main.py: Progress: %d / %d' % (env.now * 1e-9, parameters.SIM_TIME * 1e-9))
		yield env.timeout(1e9)

@profile
def save_NodeTrajectories(file_path: str):
	os.makedirs(file_path, exist_ok=True)
	# TRAJECTORY_FOLDER
	for node in range(parameters.NUMBER_OF_NODES):
		np.savetxt(file_path+"{}Node{}.txt".format(parameters.MOBILITY_MODEL,node), parameters.Node_Location_Waypoint[node], delimiter=",", fmt="%.2f")


@profile
def parse_args() -> tuple[argparse.Namespace, dict[str, str]]:
	parser = argparse.ArgumentParser()

	parser.add_argument("seed", type=int, help="Seed value for run")
	parser.add_argument("--no-input", action="store_true",
							help="Disable interactive prompts")
	
	# DEFAULT Locations of config and schema unless paths are passed in using --config=[CONFIG_PATH] or --schema-[SCHEMA_PATH]
	parser.add_argument("--config", type=str, help="Path to the config file to use", default="config/cfg/default.yaml")
	parser.add_argument("--schema", type=str, help="Path to the schema file to use", default="config/schemas/schema.json")

	known_args, unknown_args = parser.parse_known_args()

	overrides: dict[str, str] = {}

	for arg in unknown_args:
		if arg.startswith("--") and "=" in arg:
			key, value = arg[2:].split("=", 1)
			overrides[key] = value
		else:
			parser.error(f"Invalid argument format: {arg}")

	return known_args, overrides

@profile
def infer_type_and_convert(value: str, target_type: type[T] | Any | str) -> T | Any:
	try:
		if target_type == bool:
			return value.lower() in ("1", "true", "yes")
		elif target_type == list:
			return ast.literal_eval(value)
		elif getattr(target_type, '__origin__', None) is list:
			inner_type = target_type.__args__[0] # type: ignore
			parsed = ast.literal_eval(value)
			return [inner_type(item) for item in parsed]
		elif getattr(target_type, '__module__', None) == 'numpy' and target_type.__name__ == 'ndarray': # type: ignore
			parsed = ast.literal_eval(value)
			return np.array(parsed)
		return target_type(value) # type: ignore
	except Exception as _e:
		return value

@profile
def set_config_value(attr_path: str, value: str) -> None:
	parts = attr_path.split(".")
	module_var = parameters

	for part in parts[:-1]:
		module_var = getattr(module_var, part)

	final_attr = parts[-1]
	current_value = getattr(module_var, final_attr)

	inferred_type: type[Any] | str | Any = type(current_value) # type: ignore[reportUnknownVariableType]
	if is_dataclass(module_var):
		for f in fields(module_var):
			if f.name == final_attr:
				inferred_type = f.type
				break

	converted = infer_type_and_convert(value, inferred_type)
	setattr(module_var, final_attr, converted)


@dataclass(slots=True, frozen=True)
class AskNode:
	path: str
	prompt: str
	when: dict[str, Any]
	default: Any
	type_hint: str
	deps: set[str]
	predicate: Callable[[dict[str, Any], Any, set[str]], bool]

def get_from_parameters(path: str) -> Any | None:
	cur: Any = parameters
	for part in path.split("."):
		try:
			cur = cur[part] if isinstance(cur, dict) else attrgetter(part)(cur) # type: ignore
		except (KeyError, AttributeError):
			return None
	return cur # type: ignore


def compile_predicate(when: dict[str, Any]) -> Callable[[dict[str, Any], Any, set[str]], bool]:
	def predicate(values: dict[str, Any], parameters: Any, askable_set: set[str]) -> bool:
		for path, expected in when.items():
			if path in values:
				val = values[path]
			elif path in askable_set:
				# print(f"[INPUT] Condition for {path} not yet satisfied")
				return False
			else:
				val = get_from_parameters(parameters, path) # type: ignore

			if val != infer_type_and_convert(expected, type(val)): # type: ignore
				# print(f"[INPUT] Condition failed for {path}: ", f"expected {expected}, got {val}")
				return False
		return True

	return predicate

def collect_asknodes(schema: Mapping[str, Any], prefix: str = "") -> dict[str, AskNode]:
	nodes: dict[str, AskNode] = {}

	for key, value in schema.items():
		path = f"{prefix}.{key}" if prefix else key

		if not isinstance(value, MappingABC):
			continue

		if "ask_user" in value:
			when: dict[str, Any] = value.get("ask_user_when", {}) # type: ignore
			nodes[path] = AskNode(path=path, prompt=value["ask_user"], when=when, default=value.get("default", ""), type_hint=value.get("type-hint", "str"),  # type: ignore
								deps=set(when), predicate=compile_predicate(when)) # type: ignore
			# print(f"[INPUT] Found askable: {path}")

		nodes |= collect_asknodes(value, path) # type: ignore

	return nodes

def topo_sort(nodes: dict[str, AskNode]) -> list[str]:
	ts: TopologicalSorter[str] = TopologicalSorter()

	for node in nodes.values():
		ts.add(node.path, *(d for d in node.deps if d in nodes))

	try:
		order = list(ts.static_order())
	except CycleError:
		raise RuntimeError("[INPUT] Cycle detected in ask_user_when")

	# print(f"[INPUT] No cycles detected.")
	return order

def apply_interactive(schema: Mapping[str, Any]) -> dict[str, Any]:
	nodes = collect_asknodes(schema)
	order = topo_sort(nodes)

	values: dict[str, Any] = {}
	askable_set: set[str] = set(nodes)

	for path in order:
		node = nodes[path]

		if path in values:
			continue

		if node.when and not node.predicate(values, parameters, askable_set):
			# print(f"[INPUT] Skipping {path} because condition not met")
			continue

		default = (node.default.get("eval") if isinstance(node.default, dict) else node.default) # type: ignore

		resp = prompt(node.prompt + "\n", default=str(default)) # type: ignore
		values[path] = infer_type_and_convert(resp, cast(type, eval(node.type_hint)))

		# print(f"[INPUT] User set {path} = {values[path]}")

	return values

def build_overrides(values: dict[str, Any]) -> dict[str, Any]:
	root: dict[str, Any] = {}

	for path, val in values.items():
		cur: dict[str, Any] = root
		*parents, leaf = path.split(".")
		for p in parents:
			cur = cur.setdefault(p, {})
		cur[leaf] = {"eval": str(val)}

	return root

if __name__ == '__main__':
	
	# Get args excluding filename
	total_args = len(sys.argv) -1
	args, overrides = parse_args()

	if (total_args == 0):
		print("ERROR NO input argument to main.py")
		# random.seed(7447)
		# np.random.seed(random_seed)
		# main()
		exit()
	
	parameters.init_parameters(config_path=args.config, schema_path=args.schema, only_load_configs=True) # Only loading configs so we know the types for interactive input
	
	loaded_schema: dict[str, Any] = {}
	with open(Path(args.schema).with_suffix(".yaml").resolve()) as schema_file:
		loaded_schema = yaml.safe_load(schema_file)
		loaded_schema = loaded_schema.get("Config", {})

	flat_overrides = {k: infer_type_and_convert(v, type(get_from_parameters(k))) for k, v in overrides.items()} # type: ignore

	if not args.no_input:
		flat_responses = apply_interactive(loaded_schema)
		# Merge CLI overrides, but only for keys not already set by user
		for k, v in flat_overrides.items(): # type: ignore
			if k not in flat_responses:
				flat_responses[k] = v
		parameters.Interactive_Responses = {"Config": build_overrides(flat_responses)}
	else:
		parameters.Interactive_Responses = {"Config": build_overrides(flat_overrides)} # type: ignore

	parameters.init_parameters(config_path=args.config, schema_path=args.schema)

	if overrides:
		for key, val in overrides.items():
			try:
				set_config_value(key, val)
			except Exception as e:
				print(f"Failed to set {key}: {e}")

	random_seed = args.seed
	
	random.seed(random_seed)
	np.random.seed(random_seed) #SHREY*: Added np.random.seed(random_seed)

	traffic_flows = len(parameters.MOBILITY.TARGET_LOCATION) if parameters.MOBILE_SCENARIO else len(parameters.FORCED_ROUTES)

	print("Seed:", random_seed)
	print("Traffic flows:", traffic_flows)
	print("Trajectory Run No./ Mobility Model :", parameters.MOBILITY_MODEL) #SHREY* used in file path to save and read node trajectories

	# Sanity check conditions
	if (len(parameters.FORCED_ROUTES) * 2 > parameters.NUMBER_OF_NODES) or (len(parameters.MOBILITY.TARGET_LOCATION) > parameters.NUMBER_OF_NODES):
		print("Number of source + destination nodes are more than total available nodes in the network. Fix it!")
		exit()
	
	# elif (not (parameters.MOBILITY_MODEL in ["4GM_","5GM_","13GM_","14GM_", "BSCAP_"])):
	elif (not (parameters.MOBILITY_MODEL in ["BSCAP_", "TCPIPE_BSCAP_"])):
		print("Incorrect trajectory run number / mobility model. Check input! Expecting: MOBILITYMODEL_")
		exit()

	# SD: parameters.MOBILITY.TC_PIPE_mobility_ON is set to True, if parameters.MOBILITY_MODEL = "TCPIPE_BSCAP_" obtained from input sys.args[2]
	if parameters.MOBILITY_MODEL == "TCPIPE_BSCAP_":
		parameters.MOBILITY.TC_PIPE_mobility_ON = True
	elif parameters.MOBILITY_MODEL == "BSCAP_":
		parameters.MOBILITY.TC_PIPE_mobility_ON = False
	else:
		print("Incorrect mobility model name. Check input! Expecting: TCPIPE_BSCAP_ or BSCAP_")
		exit()
		
	# # Select unique source-destination pairs
	# picked_numbers = random.sample(list(range(0, parameters.NUMBER_OF_NODES - parameters.NUMBER_OF_BS)), traffic_flows*2)   # SHREY* parameters.NUMBER_OF_NODES - 1 remove BS node from being dst
	# parameters.Routes = [picked_numbers[i:i+2] for i in range(0, len(picked_numbers), 2)]; #print(parameters.Routes)

	# # Update route details
	# parameters.Route_Details = {}
	# for eachRoute in parameters.Routes:
	#     routeIdx = len(parameters.Route_Details)
	#     parameters.Route_Details[routeIdx] = {'Route': eachRoute, 
	#                                           'MCS_Index': [parameters.DEFAULT_DATA_MCS_INDEX for _ in range(len(eachRoute)-1)], 
	#                                           'Priority': 1+routeIdx if parameters.USE_FLOW_PRIORITY else 1,
	#                                           'ValidRoute': False, 
	#                                           'HQRoute': False,
	#                                           'RouteNodeDensity':[None for _ in range(len(eachRoute))]}
	# print("Seed value:", random_seed, "Route initialization: ",parameters.Route_Details)
	
	print("Seed value:", random_seed)
	print("Mobility Model:", parameters.MOBILITY_MODEL)
	print("Number of Nodes: {} UAVs + {} BS; Node Speed: {} m/s".format(parameters.NUMBER_OF_NODES-parameters.NUMBER_OF_BS, parameters.NUMBER_OF_BS, parameters.NODE_SPEED) )
	print("Data Rate: {} Mbps".format(parameters.PAYLOAD_DATA_RATE) )
	print("Percentage Node Failure:", parameters.MOBILITY.NODE_FAILURE_PERCENTAGE)
	print("TC_PIPE_mobility_ON:", parameters.MOBILITY.TC_PIPE_mobility_ON)
	
	# Run the code
	main()

	#save BSCAP node trajectories
	save_NodeTrajectories(parameters.TRAJECTORY_FOLDER + "Flows{}/".format(traffic_flows) ) #SHREY*: saving node trajectories
	