import atexit
from collections import defaultdict
from threading import Lock
from pathlib import Path
import simpy
import math
from scipy.constants import c, pi
import random
import pyarrow
import pyarrow.parquet as parquet
import os

import parameters

from numba import njit # type: ignore[reportUnknownVariableType]

from typing import Dict, Generator, List, Callable, Optional, ParamSpec, Tuple, TypeVar, TypedDict, TYPE_CHECKING

from simpy import Timeout



if TYPE_CHECKING:
    from phyPacket import PhyPacket
    from node import Node
    from simpy.resources.store import StorePut # type: ignore[reportMissingModuleSource]
    from simpy.events import ProcessGenerator # type: ignore[reportMissingModuleSource]


from line_profiler import profile # type: ignore[reportAssignmentType]

P = ParamSpec("P")
R = TypeVar("R")

try:
    @profile
    def check_for_line_profiler() -> None:
        pass
except:
    def profile(f: Callable[P, R]) -> Callable[P, R]:
        return f


@njit(parallel=True)  # type: ignore[reportUntypedFunctionDecorator]
def computeDistance_py(senderLatitude: float, senderLongitude: float, receiverLatitude: float, receiverLongitude: float):        
    return math.sqrt((senderLatitude - receiverLatitude) ** 2 + (senderLongitude - receiverLongitude) ** 2)

@njit(parallel=True)  # type: ignore[reportUntypedFunctionDecorator]
def computeDistanceNoSqrt_py(senderLatitude: float, senderLongitude: float, receiverLatitude: float, receiverLongitude: float):        
    return (senderLatitude - receiverLatitude) ** 2 + (senderLongitude - receiverLongitude) ** 2

try:
    from phero_c import computeDistance as computeDistance_cy
    from phero_c import computeDistanceNoSqrt as computeDistanceNoSqrt_cy
    computeDistance = computeDistance_cy
    computeDistanceNoSqrt = computeDistanceNoSqrt_cy
except:
    print("ether.py: error loading cython file; defining in pure python (slower)")
    print("  to build, run:  python3 setup.py build_ext --inplace ")
    computeDistance = computeDistance_py
    computeDistanceNoSqrt = computeDistanceNoSqrt_py


TransmitRow = TypedDict("TransmitRow", {"Time": int, "Tx": float, "ID": str, "INTF": float, "Dist": float, "MCS": int})
TRANSMIT_CACHE: Dict[str, List[TransmitRow]] = defaultdict(list) # node name -> TransmitRow
BUFFER_LIMIT: int = 5000

FLUSH_LOCKS: dict[str, Lock] = defaultdict(Lock)

# Seems like PyArrow is working on a new version with Typing which will be released with version 23.0.0
# Currently using 22.0.0, "# type: ignore" will likely be unnecessary when PyArrow updates
def flush_to_intermediate(src: str):
    global TRANSMIT_CACHE, FLUSH_LOCKS
    
    with FLUSH_LOCKS[src]:
        rows = TRANSMIT_CACHE.get(src)
        if not rows:
            return

        table: "pyarrow.Table" = pyarrow.Table.from_pylist(rows) # type: ignore
        intermediate_path = Path(parameters.RESULTS_FOLDER) / f"transmit_{src}_intermediate.parquet"

        if os.path.exists(intermediate_path):
            existing: "pyarrow.Table" = parquet.read_table(intermediate_path) # type: ignore
            combined: "pyarrow.Table" = pyarrow.concat_tables([existing, table]) # type: ignore
            parquet.write_table(combined, intermediate_path, compression="snappy") # type: ignore
        else:
            parquet.write_table(table, intermediate_path, compression="snappy") # type: ignore

        TRANSMIT_CACHE[src].clear()

def finalize_transmission_logs():
    global TRANSMIT_CACHE, FLUSH_LOCKS
    # Flush all caches
    for src in list(TRANSMIT_CACHE.keys()):
        flush_to_intermediate(src)

    # Rewrite each file in highest compression
    for src in TRANSMIT_CACHE.keys():
        with FLUSH_LOCKS[src]:
            intermediate_path = f"{parameters.RESULTS_FOLDER}/transmit_{src}_intermediate.parquet"
            final_path = f"{parameters.RESULTS_FOLDER}/transmit_{src}.parquet"

            if os.path.exists(intermediate_path):
                table = parquet.read_table(intermediate_path) # type: ignore
                parquet.write_table( # type: ignore
                    table, # type: ignore
                    final_path,
                    compression="zstd",
                    compression_level=22,
                    row_group_size=512 * 1024 * 1024,
                )
                os.remove(intermediate_path)

atexit.register(finalize_transmission_logs)

class Ether(object):
    @profile
    def __init__(self, env: simpy.Environment, capacity: float = float('inf')) -> None:
        self.env: simpy.Environment = env
        self.capacity: float = capacity
        self.channelsAndListeningNodes: List[Tuple[simpy.Store, Node]] = []
        self.receivingPowerFactor = pow(parameters.WAVELENGTH/(4 * pi), 2)
    
    @profile
    def log_transmission(self, phyPkt: 'PhyPacket', retransmit_number: int, received_power_current_timeslot: float) -> None:
        global TRANSMIT_CACHE, BUFFER_LIMIT
        
        src = phyPkt.macPkt.source
        
        src_node: Node = parameters.NODE_REGISTRY[src]
        dst_node: Node = parameters.NODE_REGISTRY[phyPkt.macPkt.destination]

        dist = computeDistance(
            src_node.latitude, src_node.longitude,
            dst_node.latitude, dst_node.longitude
        )

        TRANSMIT_CACHE[src].append({
            "Time": int(self.env.now),
            "Tx": float(phyPkt.macPkt.tx_power),
            "ID": str(phyPkt.macPkt.id) + ('ACK' if phyPkt.macPkt.ack else '') + str(retransmit_number),
            "INTF": float(received_power_current_timeslot),
            "Dist": float(dist),
            "MCS": int(phyPkt.macPkt.MCS_index),
        })

        if len(TRANSMIT_CACHE[src]) >= BUFFER_LIMIT:
            flush_to_intermediate(src)
        
    
    # ANDRES: changed recievingPower line so that the power used is the one in the mac packet instead of the constant parameters.Transmitting_Power, since the mac packet value is going to be the transmission power of the node (not the routing power)
    @profile
    def latencyAndAttenuation(self, phyPkt: 'PhyPacket', sourceLatitude: float, sourceLongitude: float, destinationChannel: simpy.Store, destinationNode: 'Node', beginOfPacket: bool, endOfPacket: bool) -> Generator[Timeout, None, 'StorePut']:
        #print("ether.py: latency and attenutation", sourceLatitude, sourceLongitude, destinationNode.name, beginOfPacket, endOfPacket)
        distanceNoSqrt = computeDistanceNoSqrt(sourceLatitude, sourceLongitude, destinationNode.latitude, destinationNode.longitude) + 1e-12 # add 1um to avoid distance=0
        delay = round((math.sqrt(distanceNoSqrt) / c) * 1_000_000_000, 0)
        yield self.env.timeout(delay)
        receivingPower = phyPkt.macPkt.tx_power * self.receivingPowerFactor / (distanceNoSqrt) # NB. used FSPL propagation model with isotropic antennas
        phyPkt.power[destinationNode.name] = receivingPower

        if endOfPacket:
            if int(random.random() * 101) < parameters.PACKET_LOSS_RATE * 100: # AT: int(random.random() * Z) generates in the range [0, Z) faster than randint
                phyPkt.corrupted = True

        return destinationChannel.put((phyPkt, beginOfPacket, endOfPacket))

    # ANDRES: changed if condition so that it uses the distance corresponding to the mac transmission power and not parameters.Transmitting_Power
    @profile
    def transmit(self, phyPkt: 'PhyPacket', sourceLatitude: float, sourceLongitude: float, beginOfPacket: bool, endOfPacket: bool, received_power_current_timeslot: float, retransmit_number: int, dest_channels: Optional[List[simpy.Store]] = None) -> Tuple[simpy.AllOf, List[simpy.Store]]:
        #events = [self.env.process(self.latencyAndAttenuation(phyPkt, sourceLatitude, sourceLongitude, destinationChannel, destinationNode, beginOfPacket, endOfPacket)) for destinationChannel, destinationNode in zip(self.channels, self.listeningNodes)]
        # Sh: Create events only for those nodes that are within the receiver sensitivity range, instead of all nodes in the network. Keep using them until the end of packet.
        '''events = [self.env.process(self.latencyAndAttenuation(phyPkt, sourceLatitude, sourceLongitude, destinationChannel, destinationNode, beginOfPacket, endOfPacket)) \
                            for destinationChannel, destinationNode in zip(self.channels, self.listeningNodes) \
                            if (computeDistance(sourceLatitude, sourceLongitude, destinationNode.latitude, destinationNode.longitude) <= parameters.MAX_Transmission_Range)]
                            #parameters.MAX_TX_RANGE_FOR_RX_SENSITIVITY)]'''
        destination_channels_within_range: List[simpy.Store] = []
        events: List[simpy.Process] = []
        
        if not phyPkt.macPkt.ack:
            self.log_transmission(phyPkt, retransmit_number, received_power_current_timeslot)
        
        # Currently sending events to all other nodes, to send to just some refer to below code
        # distance = computeDistance(sourceLatitude, sourceLongitude, destinationNode.latitude, destinationNode.longitude)
        # if ((distance <= parameters.txRangeAtInterferenceLevel(phyPkt.macPkt.tx_power)) or ((destChannels is not None) and (destinationChannel in destChannels))):    
        
        # Iterate over the channels and listeningNodes
        for destinationChannel, destinationNode in self.channelsAndListeningNodes:
            if True:
                # Append the event to the list
                event = self.env.process(self.latencyAndAttenuation(phyPkt, sourceLatitude, sourceLongitude, destinationChannel, destinationNode, beginOfPacket, endOfPacket))
                events.append(event)
                destination_channels_within_range.append(destinationChannel)
                if parameters.PRINT_LOGS:
                    print(f"ether.py: @{self.env.now} sending packet {phyPkt.macPkt.id} to neighbor {destinationNode.name}. begin {beginOfPacket} or end {endOfPacket}")
        return self.env.all_of(events), destination_channels_within_range

    @profile
    def getInChannel(self, node: 'Node') -> simpy.Store:
        channel = simpy.Store(self.env, capacity=self.capacity)
        self.channelsAndListeningNodes.append((channel, node))
        return channel
    
    @profile
    def removeInChannel(self, inChannel: simpy.Store, node: 'Node') -> None:
        self.channelsAndListeningNodes.remove((inChannel, node))
