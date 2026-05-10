from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Callable, Dict, List, Optional, ParamSpec, Tuple, TypeVar
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std

import parameters

# Sh
import csv
import atexit
from RouteFunctions import getFlowDest, getNodeIdx, getFlowSrc, getFlowDestStat, getRoute


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

@dataclass(slots=True)
class QueueEntry:
    packet_id: str
    enqueue_time: float
    dequeue_time: float
    dropped: int
    drop_reason: str
    freezes: int
    tx_intf: float
    rx_intf: float
    
    @profile
    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))
        

QCOLS = ["Packet ID","Enqueue Time (in s)","Dequeue Time (in s)","Dropped?", "Drop Reason", "Freezes", "Tx INTF", "Rx INTF"]
QCACHE: Dict[str, Dict[str, QueueEntry]] = defaultdict(dict)

@dataclass(slots=True)
class FlowEntry:
    packet_id: str
    route: List[int]
    generation_time: float
    first_tx_at: float
    delay: float
    times_packet_forwarded: int
    retransmissions: int
    packet_size: float
    tx_power: str
    avg_tx_power_in_route: float
    min_sinr_along_route: float
    sinr_at_flow_dest: float
    min_data_rate_along_route: float
    data_rate_at_flow_src: float
    
    @profile
    def __iter__(self):
        return (getattr(self, f.name) for f in fields(self))

FCOLS = ["Pkt ID", "Route", "Generation Time (in s)", "First Tx At (in s)", "Delay (in s)", "Times Packet Forwarded", "Retransmissions",
            "Packet Size (in bytes)", "Tx Power (in watt)", "Average Transmission Power in the route (in watt)", "Min. SINR Along Route",
            "SINR at Flow Dest","Min. Data Rate Along Route (in Mbps)","Data Rate at Flow Src (in Mbps)"]
FCACHE: Dict[str, Dict[str, FlowEntry]] = defaultdict(dict)

@profile
def dump_to_disk():
    for nodeName in QCACHE:
        with open(parameters.RESULTS_FOLDER+nodeName+".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(QCOLS)
            writer.writerows(QCACHE[nodeName].values())
    for flowName in FCACHE:
        with open(parameters.RESULTS_FOLDER+flowName, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(FCOLS)
            writer.writerows(FCACHE[flowName].values())

atexit.register(dump_to_disk)


class Stats(object):
    @profile
    def __init__(self):
        if parameters.PRINT_LOGS:   # Sh
            print("stats.py: Initialization")
        self.generatedPacketsTimes: Dict[str, float] = {}     # packet id - timestamp of generation
        self.deliveredPacketsTimes: Dict[str, float] = {}    # packet id - timestamp of delivery
        self.retransmissionTimes: List[float] = []   # timestamps of retransmissions

    # Sh: Create flow id
    @profile
    def createFlowID(self, nodeA: str, nodeB: str) -> str:
        return "Flow_"+nodeA+"-"+nodeB+".csv"

    # Sh: Get hop-based and flow-based packet IDs
    @profile
    def packetIDs(self, pktID: str) -> Tuple[str, str]:
        # Handle combined packets
        pkt_ids = pktID.split('_COMBINED_')
        is_combined = len(pkt_ids) > 1
        if is_combined:
            # For combined packets, use the first sub-packet ID to get hopBasedID and flowBasedID
            # All sub-packets in a combined packet should have the same hohopBasedIDp and but have unique flowBasedID
            pktID = pkt_ids[0]    # Use the first sub-packet ID to get hopBasedID
            parts = pktID.split('_')
            hopBasedID = '_'.join(parts[:3])
            
            temp: List[str] = []
            for pkt_id in pkt_ids:
                parts = pkt_id.split('_')
                flowBasedID = '_'.join(parts[3:])
                temp.append(flowBasedID)
            flowBasedID = '_COMBINED_'.join(temp) # flowBasedID is a string of all the flowBasedIDs of the sub-packets separated by '_COMBINED_'
        else:
            # For non-combined packets, get hopBasedID and flowBasedID directly
            parts = pktID.split('_')
            hopBasedID = '_'.join(parts[:3])
            flowBasedID = '_'.join(parts[3:])
        # print("stats.py: ", hopBasedID, flowBasedID)
        return hopBasedID, flowBasedID

    @profile
    def isPacketNew(self, nodeName: str, pkt_id: str):
        pkt_ids = pkt_id.split('_COMBINED_')
        
        for sub_pkt_id in pkt_ids:
            if sub_pkt_id in QCACHE[nodeName]:
                return False
        return True

    # Sh: Returns True when a packet entry has been successfully updated in the file. Return False when the packet id is not inserted.
    @profile
    def loqQueueStats(self, nodeName: str, pkt_id: str, retransmit_cnt: int, pkt_enqueue_time: float, pkt_dequeue_time: float, freeze_counter: int, tx_intf: float, rx_intf: float, forwarded_or_dropped: int = -1, reason_for_drop: Optional[str] = None, write_new: bool = False) -> bool:
        reason_for_drop = parameters.WAITING_FOR_TX if reason_for_drop is None else reason_for_drop
        
        pkt_ids = pkt_id.split('_COMBINED_')
        # is_combined = len(pkt_ids) > 1
        if parameters.PRINT_LOGS:
            print(f"stats.py: Dequeuing packetid(FlowBasedID): {pkt_id} and sub-packets(FlowBasedID): {pkt_ids}")
        
        for sub_pkt_id in pkt_ids:
            # sub_pkt_id = f"{sub_pkt_id}{retransmit_cnt}"
            # Packet is enqueued
            if (forwarded_or_dropped == -1 or write_new):
                
                if sub_pkt_id in QCACHE[nodeName]:
                    print(f"DUPE: {sub_pkt_id}, {nodeName} id {pkt_id} was a dupe; retransmit: {retransmit_cnt}, {pkt_enqueue_time=}, {pkt_dequeue_time=}, {forwarded_or_dropped=}, {reason_for_drop=}")
                    return False
                
                QCACHE[nodeName][sub_pkt_id] = QueueEntry(
                    sub_pkt_id,                            # Packet ID
                    round(pkt_enqueue_time/(1e9), 6),    # Packet enqueue time
                    round(pkt_dequeue_time/(1e9), 6),    # Packet dequeue time
                    forwarded_or_dropped,               # Packet forwarded successfully or dropped
                    reason_for_drop,                     # Reason for packet drop
                    0 if write_new and forwarded_or_dropped != -1 else -freeze_counter, # If this is a complete entry (there will be no dequeue call), store 0
                    tx_intf,                                                          # Else, E.g. t=0, counter=5 so we record -5, then at t=10, counter=15, below we add 15+(-5) = 10 which is the difference in freeze counter between t=0 and t=10
                    rx_intf
                )
            # Packet is dequeued
            else:
                if sub_pkt_id in QCACHE[nodeName]:
                    
                    if (forwarded_or_dropped == parameters.PKT_FORWARDED_SUCCESSFULLY):
                        reason_for_drop = ''
                    
                    old_data = QCACHE[nodeName][sub_pkt_id]
                    
                    if old_data.freezes > 0:
                        print(f"DUPE: {sub_pkt_id}, {nodeName} id {pkt_id} was a dupe; retransmit: {retransmit_cnt}, {pkt_enqueue_time=}, {pkt_dequeue_time=}, {forwarded_or_dropped=}, {reason_for_drop=}")
                        return False
                    
                    old_data.dequeue_time = round(pkt_dequeue_time/(1e9), 6)
                    old_data.dropped = forwarded_or_dropped
                    old_data.drop_reason = reason_for_drop
                    old_data.freezes = freeze_counter + old_data.freezes
                    old_data.tx_intf = tx_intf
                    old_data.rx_intf = rx_intf
                    if parameters.PRINT_LOGS: 
                        # Debug print for packet dequeue
                        print(f"stats.py: Dequeuing packet {sub_pkt_id} with drop reason: {reason_for_drop}")
                    
                    # Update node's PST statistic
                    nodeId = getNodeIdx(nodeName)
                    curr_tot_pkts = int(parameters.Node_Stats[nodeId]['PktsServed'])
                    curr_PST = float(parameters.Node_Stats[nodeId]['PST'])
                    pst = (float(old_data.dequeue_time) - float(old_data.enqueue_time))
                    # Formula: Updated Avg PST = (Curr PST * Curr Total Pkts Served + PST of this Pkt)/(Curr Total Pkts Served + 1)
                    parameters.Node_Stats[nodeId]['PST'] = round((curr_PST*curr_tot_pkts + pst) / (1+curr_tot_pkts), 6)
                    parameters.Node_Stats[nodeId]['PktsServed'] = 1+curr_tot_pkts
        return True

    # ANDRES: Added new parameter transmission power since before it used directy parameters.Transmitting_Power
    @profile
    def logTransmittedPacket(self, id: str, timestamp: float, length: int, mcs_index: int, transmission_power: float, pktOriginatedBySrc: bool) -> None:    # Sh: Changed function name and added extra fields
        # Check if this is a combined packet
        pkt_ids = id.split('_COMBINED_')
        is_combined = len(pkt_ids) > 1
        # print("ORGINAL ID:", id,"pkt_ids:", pkt_ids, "is_combined:", is_combined")
        # For combined packets, split the length equally
        if is_combined:
            length = length // 2
        
        for pkt_id in pkt_ids:
            
            # Sh: Get hop and flow based packet IDs
            hopBasedID, flowBasedID = self.packetIDs(pkt_id)
            self.generatedPacketsTimes[hopBasedID] = timestamp  # Sh: Modified. Earlier the index was id

            # Sh: Log flow stats
            flowSrc = getFlowSrc(flowBasedID)
            flowDst = getFlowDest(flowBasedID)
            flowName = self.createFlowID(flowSrc, getFlowDestStat(flowBasedID))
            dataRate = parameters.MCS_SNR_TABLE[mcs_index]['DataRate']
            # Create a new entry if flow source originated this packet
            if (pktOriginatedBySrc):
                FCACHE[flowName][flowBasedID] = FlowEntry(
                                flowBasedID,                        # Packet ID
                                getRoute(flowSrc, flowDst),           # Route
                                round((float(flowBasedID.split('_')[0])/1e9), 6),   # Packet generation time
                                round(timestamp/(1e9), 6),               # Time at first transmission
                                float("inf"),                    # Delay
                                0,                                      # Times intermediate node(s) forwarded this packet.
                                0,                                      # Retransmissions
                                length/8,                               # Packet size
                                str(transmission_power),                # Tx power // ANDRES: now is a string since we are adding each node's txPower
                                transmission_power,                     # Average Tx powers // ANDRES: new column, calculates the average txPower along the current route
                                float("inf"),                    # Min. SINR along route
                                float("inf"),                    # SINR at flow destination
                                round(dataRate, 1),                      # Min. data rate along route
                                round(dataRate, 1)                       # Data rate at flow source
                )
            # Intermediate node is attempting to forward this packet. Update stats
            else:
                if flowBasedID in FCACHE[flowName]:
                    oldRate = FCACHE[flowName][flowBasedID].min_data_rate_along_route
                    FCACHE[flowName][flowBasedID].min_data_rate_along_route = round(min(dataRate, oldRate), 1)

                    # ANDRES: Added that now every transmission power of every node in the route is added and new column for the average along the route
                    current_txPowers = FCACHE[flowName][flowBasedID].tx_power
                    new_txPowers = str(current_txPowers) + "," + str(transmission_power)
                    FCACHE[flowName][flowBasedID].tx_power = new_txPowers

                    txPowers_values = new_txPowers.split(",")
                    txPowers_values_float = [float(value) for value in txPowers_values]

                    FCACHE[flowName][flowBasedID].avg_tx_power_in_route = round(sum(txPowers_values_float) / float(len(txPowers_values_float)), 6)
                else:
                    raise Exception(f"{id}|{pkt_id} {timestamp} delivered but {flowBasedID} not in cache")
    
    @profile
    def logDeliveredPacket(self, id: str, timestamp: float, pktReachedFlowDst: bool, sinr: float) -> None:   # Sh: Added extra field
        pkt_ids = id.split('_COMBINED_')
        
        for pkt_id in pkt_ids:
            hopBasedID, flowBasedID = self.packetIDs(pkt_id)
            
            self.deliveredPacketsTimes[hopBasedID] = timestamp  # Sh: Modified. Earlier the index was id

            # Sh: Update (i) min. SINR along route, and (ii) delay and SINR at flow destination if the packet reached the flow destination node
            flowSrc = getFlowSrc(flowBasedID)
            # flowDst = getFlowDest(flowBasedID)
            flowName = self.createFlowID(flowSrc, getFlowDestStat(flowBasedID))
            if flowBasedID in FCACHE[flowName]:
                prevSINR = FCACHE[flowName][flowBasedID].min_sinr_along_route
                FCACHE[flowName][flowBasedID].min_sinr_along_route = round(min(sinr, prevSINR), 6)
                if (pktReachedFlowDst):
                    FCACHE[flowName][flowBasedID].delay = round(timestamp/(1e9) - FCACHE[flowName][flowBasedID].generation_time, 6)   # Packet's end-to-end delay
                    FCACHE[flowName][flowBasedID].sinr_at_flow_dest = round(sinr, 6)
            else:
                raise Exception(f"{id} {timestamp} delivered but {flowBasedID} not in cache")

    @profile
    def logSuccessfulForward(self, id: str, timestamp: float) -> None:
        pkt_ids = id.split('_COMBINED_')
        
        for pkt_id in pkt_ids:
            _, flowBasedID = self.packetIDs(pkt_id)
            
            flowSrc = getFlowSrc(flowBasedID)
            flowName = self.createFlowID(flowSrc, getFlowDestStat(flowBasedID))
            if flowBasedID in FCACHE[flowName]:
                FCACHE[flowName][flowBasedID].times_packet_forwarded += 1
            else:
                raise Exception(f"{id} {timestamp} forwarded but {flowBasedID} not in cache")

    @profile
    def logRetransmission(self, timestamp: float, id: str) -> None: # Sh: Added extra field
        self.retransmissionTimes.append(timestamp)

        # Sh: Update total retransmission count along the route for this packet
        _, flowBasedID = self.packetIDs(id)
        flowSrc = getFlowSrc(flowBasedID)
        # flowDst = getFlowDest(flowBasedID)
        flowName = self.createFlowID(flowSrc, getFlowDestStat(flowBasedID))
        
        if flowBasedID in FCACHE[flowName]:
            FCACHE[flowName][flowBasedID].retransmissions += 1
        else:
            raise Exception(f"{id} {timestamp} retransmission but {flowBasedID} not in cache")
        

    @profile
    def printGeneratedPacketTimes(self):
        for generatedPacket in self.generatedPacketsTimes:
            print (self.generatedPacketsTimes[generatedPacket])

    @profile
    def printDeliveredPacketTimes(self):
        for deliveredPacket in self.deliveredPacketsTimes:
            print (self.deliveredPacketsTimes[deliveredPacket])

    @profile
    def plotCumulativePackets(self) -> None:
        figure: Figure
        ax: Axes
        figure, ax = plt.subplots()  # type: ignore[reportUnknownMemberType]

        cumulativeGeneratedPackets: List[int] = [1]
        generatedPacketsTimes: List[float] = []
        i = 0
        for packet in self.generatedPacketsTimes:
            if i != 0:
                cumulativeGeneratedPackets.append(cumulativeGeneratedPackets[i-1] + 1)
            generatedPacketsTimes.append(self.generatedPacketsTimes[packet] * 1e-9)
            i += 1

        cumulativeDeliveredPackets: List[int] = [1]
        deliveredPacketsTimes: List[float] = []
        i = 0
        for packet in self.deliveredPacketsTimes:
            if i != 0:
                cumulativeDeliveredPackets.append(cumulativeDeliveredPackets[i-1] + 1)
            deliveredPacketsTimes.append(self.deliveredPacketsTimes[packet] * 1e-9)
            i += 1

        ax.plot(generatedPacketsTimes, cumulativeGeneratedPackets, 'r:', label='Generated') # type: ignore[reportUnknownMemberType]
        ax.plot(deliveredPacketsTimes, cumulativeDeliveredPackets, 'g:', label='Delivered') # type: ignore[reportUnknownMemberType]

        ax.set_xlabel('Time (s)') # type: ignore[reportUnknownMemberType]
        ax.set_ylabel('Packets') # type: ignore[reportUnknownMemberType]
        ax.legend() # type: ignore[reportUnknownMemberType]
        
        #file = 'results/packets' + str(parameters.TARGET_PKT_GENERATION_RATE) + '.pdf'
        file = parameters.RESULTS_FOLDER + 'packets' + str(parameters.TARGET_PKT_GENERATION_RATE[0]) + '.png' # Sh
        figure.savefig(file, bbox_inches='tight', dpi=250) # type: ignore[reportUnknownMemberType]
        
        print("\nTotal number of generated packets: {}".format(len(generatedPacketsTimes)))
        print("Total number of delivered packets: {} # Use all_stats.csv to check PDR".format(len(deliveredPacketsTimes)))

        plt.close(figure)
        del cumulativeGeneratedPackets, generatedPacketsTimes, cumulativeDeliveredPackets, deliveredPacketsTimes
        

    @profile
    def plotThroughputMs(self):
        figure, ax = plt.subplots() # type: ignore[reportUnknownMemberType]
        
        packetsGeneratedEveryMillisecond: List[int] = [0] * int(parameters.SIM_TIME * 1e-6)
        packetsDeliveredEveryMillisecond: List[int] = [0] * int(parameters.SIM_TIME * 1e-6)
    
        for packet in self.generatedPacketsTimes:
            packetsGeneratedEveryMillisecond[int(self.generatedPacketsTimes[packet] * 1e-6)] += 1
    
        for packet in self.deliveredPacketsTimes:
            packetsDeliveredEveryMillisecond[int(self.deliveredPacketsTimes[packet] * 1e-6)] += 1
    
        packetsGeneratedEveryMillisecond = packetsGeneratedEveryMillisecond[int(parameters.PKT_GENERATION_START_TIME * 1e-6):]
        packetsDeliveredEveryMillisecond = packetsDeliveredEveryMillisecond[int(parameters.PKT_GENERATION_START_TIME * 1e-6):]
        milliseconds = np.arange(int(parameters.PKT_GENERATION_START_TIME * 1e-6), int(parameters.SIM_TIME * 1e-6), 1)
    
        ax.plot(milliseconds, packetsGeneratedEveryMillisecond, 'r:', label='Generated') # type: ignore[reportUnknownMemberType]
        ax.plot(milliseconds, packetsDeliveredEveryMillisecond, 'g:', label='Delivered') # type: ignore[reportUnknownMemberType]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val/1000:.0f}")) # type: ignore
        ax.hlines(mean(packetsGeneratedEveryMillisecond), int(parameters.PKT_GENERATION_START_TIME * 1e-6), milliseconds[-1], colors='black', label='Generated mean') # type: ignore[reportUnknownMemberType]
        ax.hlines(mean(packetsDeliveredEveryMillisecond), int(parameters.PKT_GENERATION_START_TIME * 1e-6), milliseconds[-1], colors='yellow', label='Delivered mean') # type: ignore[reportUnknownMemberType]
    
        ax.set_xlabel('Time (s)') # type: ignore[reportUnknownMemberType]
        ax.set_ylabel('Throughput (packets/ms)') # type: ignore[reportUnknownMemberType]
        ax.legend() # type: ignore[reportUnknownMemberType]
        
        #file = 'results/throughput' + str(parameters.TARGET_PKT_GENERATION_RATE) + '.pdf'
        file = parameters.RESULTS_FOLDER + 'throughputms' + str(parameters.TARGET_PKT_GENERATION_RATE[0]) + '.png'   # Sh
        figure.savefig(file, bbox_inches='tight', dpi=250) # type: ignore[reportUnknownMemberType]
        print("Average number of packets genetated every millisecond: {}".format(mean(packetsGeneratedEveryMillisecond)))
        print("Average number of packets delivered every millisecond: {}".format(mean(packetsDeliveredEveryMillisecond)))
        print("Standard deviation of packets genetated every millisecond: {}".format(std(packetsGeneratedEveryMillisecond)))
        print("Standard deviation of packets delivered every millisecond: {}".format(std(packetsDeliveredEveryMillisecond)))

        plt.close(figure)
        del packetsGeneratedEveryMillisecond, packetsDeliveredEveryMillisecond, milliseconds
        
    
    @profile
    def plotThroughput(self):
        figure, ax = plt.subplots() # type: ignore[reportUnknownMemberType]
        
        packetsGeneratedEverySecond = [0] * int(parameters.SIM_TIME * 1e-9)
        packetsDeliveredEverySecond = [0] * int(parameters.SIM_TIME * 1e-9)
    
        for packet in self.generatedPacketsTimes:
            packetsGeneratedEverySecond[int(self.generatedPacketsTimes[packet] * 1e-9)] += 1
    
        for packet in self.deliveredPacketsTimes:
            packetsDeliveredEverySecond[int(self.deliveredPacketsTimes[packet] * 1e-9)] += 1
    
        packetsGeneratedEverySecond = packetsGeneratedEverySecond[int(parameters.PKT_GENERATION_START_TIME * 1e-9):]
        packetsDeliveredEverySecond = packetsDeliveredEverySecond[int(parameters.PKT_GENERATION_START_TIME * 1e-9):]
        seconds = np.arange(int(parameters.PKT_GENERATION_START_TIME * 1e-9), int(parameters.SIM_TIME * 1e-9), 1)
    
        ax.plot(seconds, packetsGeneratedEverySecond, 'r:', label='Generated') # type: ignore[reportUnknownMemberType]
        ax.plot(seconds, packetsDeliveredEverySecond, 'g:', label='Delivered') # type: ignore[reportUnknownMemberType]
        ax.hlines(mean(packetsGeneratedEverySecond), int(parameters.PKT_GENERATION_START_TIME * 1e-9), seconds[-1], colors='black', label='Generated mean') # type: ignore[reportUnknownMemberType]
        ax.hlines(mean(packetsDeliveredEverySecond), int(parameters.PKT_GENERATION_START_TIME * 1e-9), seconds[-1], colors='yellow', label='Delivered mean') # type: ignore[reportUnknownMemberType]
    
        ax.set_xlabel('Time (s)') # type: ignore[reportUnknownMemberType]
        ax.set_ylabel('Throughput (packets/s)') # type: ignore[reportUnknownMemberType]
        ax.legend() # type: ignore[reportUnknownMemberType]
        
        #file = 'results/throughput' + str(parameters.TARGET_PKT_GENERATION_RATE) + '.pdf'
        file = parameters.RESULTS_FOLDER + 'throughput' + str(parameters.TARGET_PKT_GENERATION_RATE[0]) + '.png'  # Sh
        figure.savefig(file, bbox_inches='tight', dpi=250) # type: ignore[reportUnknownMemberType]
        
        print("Average number of packets genetated every second: {}".format(mean(packetsGeneratedEverySecond)))
        print("Average number of packets delivered every second: {}".format(mean(packetsDeliveredEverySecond)))
        print("Standard deviation of packets genetated every second: {}".format(std(packetsGeneratedEverySecond)))
        print("Standard deviation of packets delivered every second: {}".format(std(packetsDeliveredEverySecond)))

        plt.close(figure)
        del packetsGeneratedEverySecond, packetsDeliveredEverySecond, seconds
        

    @profile
    def plotDelays(self):
        figure, ax = plt.subplots() # type: ignore[reportUnknownMemberType]
        
        delays: List[float] = []
        deliveredPacketsTimes: List[float] = []

        for packet in self.deliveredPacketsTimes:
            deliveredPacketsTimes.append(self.deliveredPacketsTimes[packet] * 1e-6)
            delays.append(self.deliveredPacketsTimes[packet] * 1e-6 - self.generatedPacketsTimes[packet] * 1e-6)

        
        ax.plot(deliveredPacketsTimes, delays, 'b:', label='Delays') # type: ignore[reportUnknownMemberType]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val/1000:.0f}")) # type: ignore
        ax.hlines(mean(delays), int(parameters.PKT_GENERATION_START_TIME * 1e-6), deliveredPacketsTimes[-1], colors='red', label='Delays mean') # type: ignore[reportUnknownMemberType]

        ax.set_xlabel('Time (s)') # type: ignore[reportUnknownMemberType]
        ax.set_ylabel('Delay (ms)') # type: ignore[reportUnknownMemberType]
        ax.legend() # type: ignore[reportUnknownMemberType]
        
        #file = 'results/delays' + str(parameters.TARGET_PKT_GENERATION_RATE) + '.pdf'
        file = parameters.RESULTS_FOLDER + 'delays' + str(parameters.TARGET_PKT_GENERATION_RATE[0]) + '.png'  # Sh
        figure.savefig(file, bbox_inches='tight', dpi=250) # type: ignore[reportUnknownMemberType]
        
        print("Average delay: {}".format(mean(delays)))
        print("Standard deviation of delay: {}".format(std(delays)))
        print("Minimum delay: {}".format(min(delays)))
        print("Maximum delay: {}".format(max(delays)))

        plt.close(figure)
        del delays, deliveredPacketsTimes
        

    @profile
    def plotRetransmissions(self):
        figure, ax = plt.subplots() # type: ignore[reportUnknownMemberType]
        
        retransmissionsEveryMillisecond: List[int] = []
        for i in range(int(parameters.SIM_TIME * 1e-6)):
            retransmissionsEveryMillisecond.append(0)

        cumulative = 0
        for timestamp in self.retransmissionTimes:
            cumulative = cumulative + 1
            retransmissionsEveryMillisecond[int(timestamp * 1e-6)] = cumulative

        for i in range(1, len(retransmissionsEveryMillisecond)):
            if retransmissionsEveryMillisecond[i] == 0:
                retransmissionsEveryMillisecond[i] = retransmissionsEveryMillisecond[i - 1]

        retransmissionsEveryMillisecond = retransmissionsEveryMillisecond[int(parameters.PKT_GENERATION_START_TIME * 1e-6):]
        milliseconds = np.arange(int(parameters.PKT_GENERATION_START_TIME * 1e-6), int(parameters.SIM_TIME * 1e-6), 1)

        ax.plot(milliseconds, retransmissionsEveryMillisecond, 'r:', label='Retransmissions') # type: ignore[reportUnknownMemberType]
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val/1000:.0f}")) # type: ignore
        
        ax.set_xlabel('Time (s)') # type: ignore[reportUnknownMemberType]
        ax.set_ylabel('Retransmissions') # type: ignore[reportUnknownMemberType]
        ax.legend() # type: ignore[reportUnknownMemberType]
        
        #file = 'results/retransmissions' + str(parameters.TARGET_PKT_GENERATION_RATE) + '.pdf'
        file = parameters.RESULTS_FOLDER + 'retransmissions' + str(parameters.TARGET_PKT_GENERATION_RATE[0]) + '.png' # Sh
        figure.savefig(file, bbox_inches='tight', dpi=250) # type: ignore[reportUnknownMemberType]
        
        print("Total number of retransmissions: {}".format(cumulative))

        plt.close(figure)
        del retransmissionsEveryMillisecond, cumulative, milliseconds
        
