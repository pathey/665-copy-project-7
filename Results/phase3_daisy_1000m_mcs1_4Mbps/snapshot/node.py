from collections import deque
from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVar, TypedDict

import numpy as np
import simpy
import random

import parameters

from queue import Queue # Sh: To store packets in the queue in FIFO order (default mode)
from queue import PriorityQueue

from simpy.events import ProcessGenerator # MC: To store packets in the queue based on priority. Note: It may not store packets of the same priority in FIFO order.
Use_Strict_FIFO_Order = True    # Sh: Uses FIFO queue when True; priority queue when False
#Queue in Python3: https://docs.python.org/3/library/queue.html

								
from threading import Lock  # Sh: To pervent changes to the queue while its been modified
import RouteFunctions as rf # Sh: To get node id
from RouteFunctions import getFlowSrc, getNodeIdx, store_transmission_power
from stats import Stats
from ether import computeDistance

if TYPE_CHECKING:
    from ether import Ether

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

qEntry = tuple[int, str, int, str] # For non priority, ignore first value
ETD_to_destEntry = TypedDict("ETD_to_destEntry", {'Flow Id': str,'Remaining Route': list[int],'Remaining ETD': float,'Priority': int})

class Node(object):
    @profile
    def __init__(self: 'Node', env: simpy.Environment, name: str, ether: 'Ether', latitude: float, longitude: float, stats: Stats, routing_power: float, external_interference: float = 0):
        import mac
        '''if transmission_power > routing_power:
            raise ValueError(f"Invalid values for Node {name}: Transmission Power ({transmission_power}) cannot be greater than the Routing Power ({routing_power})")'''
        
        self.env: simpy.Environment = env
        self.name: str = name
        self.id: int = getNodeIdx(self.name)
        self.ether: 'Ether' = ether
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.stats: Stats = stats

        self.routing_power: float = routing_power              # ANDRES Added new value routing power that is different to the transmission power, this one is used for the routing algorithm
        
        self.operating_mcs: int = parameters.DEFAULT_DATA_MCS_INDEX

        self.interference: float = 0 # AT: 2/14/26 Keeps track of current interference observed by the node
        
        self.power: float = parameters.Transmitting_Power # AT: 2/14/26 We decided to use this power for route selection instead of self.routing_power
        
        self.tx_power_data = self.power
        self.tx_power_ack = self.power
        self.tx_power_rts = self.power
        self.tx_power_cts = self.power
        
        self.sufficient_power_to_transmit_from = {} # Key: Node Ids, Value: The power that would have been sufficient for the transmitter such that I would recieve it.
        
        self.external_interference = external_interference     # ANDRES 12/03/25: Added individual external interference for each node.

        self.txPower_increase = False
        self.packetsExpired = 0
        self.num_packets_generated: int = 0

        self.mac = mac.Mac(self)
        self.lock = Lock()  # Sh: Prevent simultaneous changes in the queue
        self.q: Queue[qEntry] | PriorityQueue[qEntry]
        if Use_Strict_FIFO_Order:   # Sh
            self.q = Queue(maxsize=parameters.MAX_QUEUE_SIZE)
        else:
            self.q = PriorityQueue(maxsize=parameters.MAX_QUEUE_SIZE)   # MC
        
        self.inKeepSending = False  # Sh: Tracks when a ndoe comes out of the keepSending()
        if not parameters.MOBILE_SCENARIO:  # Sh: Node placement in static scenario
            print('node.py: %s created with coordinates %d %d with routing power %f' % (self.name, self.latitude, self.longitude, self.routing_power))
        
        parameters.NODE_REGISTRY[self.name] = self      # ANDRES: when a node is created, we added to the registry. This way we can access a node at any point with just its name

    @profile
    def receive(self, id: str, source: str) -> None:
        if parameters.PRINT_LOGS:
            print('node.py: Time %d: %s receives %s from %s' % (self.env.now, self.name, id, source))

    # AT: Called from parameters.initialize_NODES_FLOWLOGS_QUEUE_PACKETGENERATION
    @profile
    def detect_congestion(self) -> ProcessGenerator: # Create 2 functions; one for detection, one for taking action (log: congestion, prev_mcs, current_mcs)
        if not parameters.CONGESTION_BASED_MCS: # TODO: 12/26
            return
        
        prev_packets_received_counter = 0
        prev_packets_serviced_counter = 0
        TIME_WINDOW = parameters.PACKET_TTL / 7.5  # e.g., TTL = 3s -> ~0.4s per window
        
        self_id = getNodeIdx(self.name)
        routes_with_detection = parameters.ROUTES_WITH_CONGESTION_BASED_MCS
        
        congestion_history: deque[float] = deque([], maxlen=5)
        
        while True:
            rf.updateRouteMCS() # All nodes call update before waiting for next window
            yield self.env.timeout(TIME_WINDOW)
            
            # If the node has never serviced a packet, we assume it is not part of a route or is a destination
            if self.mac.num_packets_serviced == 0:
                continue
            
            route_details = parameters.Route_Details
            if routes_with_detection and not any(((idx in routes_with_detection) and (self_id in route_details[idx]['Route'])) for idx in route_details):
                continue
            
            packets_received = (self.mac.num_packets_received + self.num_packets_generated) - prev_packets_received_counter
            packets_serviced = self.mac.num_packets_serviced - prev_packets_serviced_counter
            
            prev_packets_received_counter = (self.mac.num_packets_received + self.num_packets_generated)
            prev_packets_serviced_counter = self.mac.num_packets_serviced
            
            if packets_received == 0:
                continue
            
            congestion_ratio = packets_serviced / packets_received
            congestion_history.append(congestion_ratio)
            
            prev_mcs = self.operating_mcs
            moving_avg_congestion = np.average(congestion_history)
            
            if moving_avg_congestion <= parameters.CONGESTION_BASED_MCS_INCREMENT_THRESHOLD:
                self.operating_mcs += parameters.CONGESTION_BASED_MCS_INCREMENT_STEP
                self.operating_mcs = min(self.operating_mcs, parameters.MAX_DATA_MCS_INDEX)
            elif moving_avg_congestion >= parameters.CONGESTION_BASED_MCS_DECREMENT_THRESHOLD:
                self.operating_mcs -= parameters.CONGESTION_BASED_MCS_DECREMENT_STEP
                self.operating_mcs = max(parameters.CONTROL_MCS_INDEX, self.operating_mcs)
            
            print(f"Node.py: ({self.name})@{self.env.now}: Updated MCS from {prev_mcs} to {self.operating_mcs}, hist={congestion_history}, {packets_serviced=}, {packets_received=}")
    
    # Sh: Modified functionality. Packet Generation function (see below) calls this function.
    @profile
    def keepSending(self) -> ProcessGenerator:
        """
        Continuously sends packets from the node to its MAC layer, one at a time since MAC can only handle one packet at a time.
        Aggregates packets into a bigger packet when if it can.
        """
        if self.inKeepSending: return
        self.inKeepSending = True
        while not self.q.empty():
            # Sh: MAC can handle one packet at a time
            # print(f"TIMEOUT: {self.name}, {self.env.now}, {self.mac.hasPacket = }, {self.mac.hasACKToSend = }, {self.mac.isSensing = }")
            while self.mac.hasPacket or self.mac.hasACKToSend:# or self.mac.isSensing:# or self.mac.phy.isSending:
                yield self.env.timeout(parameters.SLOT_DURATION)
            # print(f"TIMEOUT DONE: {self.name}, {self.env.now}, {self.mac.hasPacket = }, {self.mac.hasACKToSend = }, {self.mac.isSensing = }")
            if self.q.empty():    # Sanity check
                break
            
            # if parameters.ENABLE_CONGESTION_DETECTION:
            #     # ANDRES: Checking that the queue for the packets is getting full (more than 100 packets). If that is the case, we increase the MCS index.
            #         if self.q.qsize() >= parameters.MAX_QUEUE_SIZE*0.025 and not self.txPower_increase:
            #             self.updateMCSindex(True)
            #             self.txPower_increase = True
            #     # ANDRES: Checking that the queue for the packets is emptying. If that is the case, decrease the MCS index (to default)
            #         '''if self.q.qsize() <= parameters.MAX_QUEUE_SIZE*0.01 and self.txPower_increase:
            #         self.updateMCSindex(False)
            #         self.txPower_increase = False'''

            # Extract the head-of-line packet from node's queue
            _, dstNode, pkt_length, pkt_id = self.q.get()

            srcNode = getFlowSrc(pkt_id) # Get packet (or flow) source
            nextHopNode, mcs_index, _prevHopNode, route_index = self.findNextHop(srcNode,dstNode)   # Get next hop node and MCS index
            
            # Sanity check: If no next hop node, current transmitter node preemptively drops the packet. If packet reinsertion feature is used, these packets would have been reinserted at the source node.
            if (nextHopNode is None or route_index is None):
                # Record packet drops
                if parameters.PRINT_LOGS:
                    print('node.py: Time %d: No route for Packet %s at node %s. Drop from queue!\n' % (self.env.now, pkt_id, self.name))
                # DONE: 10/23 Get interference values from table
                cur_intf_self = parameters.INTF_Registry[getNodeIdx(self.name)]
                _ = self.stats.loqQueueStats(self.name, pkt_id, 0, float(pkt_id.split('_')[0]), self.env.now, self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if cur_intf_self is None else cur_intf_self, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_NO_ROUTE)
                break
            
            # SD: Try to find another packet with the same next hop if MCS index >= 1
            if parameters.PRINT_LOGS:
                print(f"\nnode.py: Time {self.env.now}: Node {self.name} checking for Second packet to combine - MCS: {mcs_index}, FirstPacket: {pkt_id}->{nextHopNode}, Queue: {self.q.qsize()}")
            extra_packets: list[qEntry] = []
            max_packets = int(parameters.MCS_SNR_TABLE[mcs_index]['DataRate'] / parameters.MCS_SNR_TABLE[0]['DataRate']) # Note: Intentionally not DEFAULT_MCS
            max_space = max_packets * parameters.BASE_PAYLOAD_LENGTH
            while parameters.USE_PACKET_AGGREGATION and mcs_index >= 1 and not self.q.empty() and len(extra_packets) <= max_packets-1: #SD: Only for high power nodes and if MCS index >= 1
                # Calculate remaining space for combining
                if parameters.PRINT_LOGS:
                    print(f"{max_space=} - {pkt_length=},  {pkt_id}, {extra_packets}")
                if max_space - pkt_length > 0:  # Only search if we have space to combine
                    # Search queue for matching packet
                    matching_idx = None
                    for i, next_pkt in enumerate(self.q.queue):
                        next_pkt_data = next_pkt
                        _, next_dstNode, next_pkt_length, next_pkt_id = next_pkt_data
                        
                        # Early exit if packet is too large to combine
                        # print(f"{next_pkt_length=} > {max_space=} - {pkt_length=}, {next_pkt_id=}")
                        if next_pkt_length > max_space - pkt_length:
                            continue
                            
                        next_srcNode = rf.getFlowSrc(next_pkt_id)
                        next_nextHopNode, _next_mcs_index, _, _next_route_index = self.findNextHop(next_srcNode, next_dstNode)
                        
                        if parameters.PRINT_LOGS:
                            print(f"  - Checking packet {i}: {next_pkt_id}")
                            print(f"    To: {next_nextHopNode}, Src: {next_srcNode}, Dst: {next_dstNode}")
                        
                        # Check if packet matches criteria
                        if (next_nextHopNode == nextHopNode and
                            next_srcNode == srcNode and 
                            next_dstNode == dstNode):
                            matching_idx = i
                            break  # Exit as soon as we find a match
                    
                    # If matching packet found, combine them
                    if matching_idx is not None:
                        if Use_Strict_FIFO_Order:
                            extra_packets.append(self.q.queue[matching_idx]) # (Dest node, Packet Length, Packet Id)
                        else:
                            extra_packets.append(self.q.queue[matching_idx]) # (Packet Priority, (Dest node, Packet Length, Packet Id))

                        if parameters.PRINT_LOGS:
                            print(f"  - Found matching packet: {extra_packets[-1][3]}")
                            print(f"  - Combined length would be: {pkt_length + extra_packets[-1][2]}")
                        
                        # Remove the matched packet from queue using list method
                        matched_pkt = list(self.q.queue)[matching_idx]
                        self.q.queue.remove(matched_pkt)
                        pkt_length += extra_packets[-1][2]  # Combine packet lengths
                        
                        # print(f"    Packets combined successfully!")
                        assert extra_packets[-1] is not None, "Second packet should not be None after combining"
                        assert pkt_length <= parameters.BASE_PAYLOAD_LENGTH * max_packets, "Combined packet length exceeds maximum"
                    else:
                        if parameters.PRINT_LOGS:
                            print(f"    No matching packet found in queue")
                        break
                else:
                    if parameters.PRINT_LOGS:
                        print(f"    First packet too large for combining ({pkt_length} bytes)")
                    break
            if not extra_packets:
                if parameters.PRINT_LOGS:
                    print(f"    Packets not combined - {'MCS index too low' if mcs_index < 1 else 'Queue empty'}")
                pass
            # print()

            # Updated packet id: (curr time)_(transmitter node)_(next hop)_(pkt generation time)_(flow src)_(flow dst)_(No. of times reinserted)
            updated_pkt_id: str = str(round(self.env.now)) + '_' + self.name + '_' + nextHopNode + '_' + pkt_id
            for extra_pkt in extra_packets:
                updated_pkt_id += f"_COMBINED_{round(self.env.now)}_{self.name}_{nextHopNode}_{extra_pkt[3]}"  #SD: Add second packet's ID
            # print("node.py: updated_pkt_id:", updated_pkt_id)
            
            if parameters.PRINT_LOGS:
                if srcNode == self.name:    # Flow source
                    print('node.py: Time %d: %s will send %s to %s. Flow src: %s, dst: %s' % (self.env.now, self.name, updated_pkt_id, nextHopNode, srcNode, dstNode))
                else:   # Intermediate node
                    print('\nnode.py: Time %d: %s will forward %s to %s. Flow src: %s, dst: %s\n' % (self.env.now, self.name, updated_pkt_id, nextHopNode, srcNode, dstNode))
            # print("node.py: updated_pkt_id:", updated_pkt_id)
            # print("Curr time:",self.env.now, "Am i here? node:", self.name, "node status:", self.mac.hasPacket, "pkt:", updated_pkt_id)

            yield self.env.process(self.mac.send(nextHopNode, pkt_length, updated_pkt_id,1+len(extra_packets), route_index, mcs_index))   # Sh: added mcs index
        self.inKeepSending = False
        return
    
    def computePowers(self, destinationNode: str, data_mcs: int) -> None:
        # Computing Power for RTS Packet and Data Packet
        receiver_node = parameters.NODE_REGISTRY[destinationNode]
        distance_to_receiver = computeDistance(self.latitude, self.longitude, receiver_node.latitude, receiver_node.longitude)
        interference_at_receiver = parameters.INTF_Registry[getNodeIdx(destinationNode)]
        if interference_at_receiver is None: interference_at_receiver = 0

        # P_rts_default is MCS 0 at D_max + interference at the receiver
        # P_rts_datamcs is data's MCS at actual link length
        # tx_power_rts is the higher of P_rts_default and P_rts_datamcs capped at maximum transmit power
        
        # AT: 2/18/26 Adding external interference to rts data ack, previous code commented
        # P_rts_default = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=interference_at_receiver)
        # P_rts_datamcs = parameters.getTransmitPower(distance_to_receiver, mcs_index=data_mcs, interference=interference_at_receiver)
        max_tx_power = parameters.getMaximumTransmitPower(data_mcs)

        P_rts_default = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=interference_at_receiver+receiver_node.external_interference)
        P_rts_datamcs = parameters.getTransmitPower(distance_to_receiver, mcs_index=data_mcs, interference=interference_at_receiver+receiver_node.external_interference)

        self.tx_power_rts = min(max(P_rts_default, P_rts_datamcs), max_tx_power)
        self.mac.phy.tracker_power_rts.add(self.tx_power_rts)
        
        # tx_power_data is at data MCS at actual link length + interference at receiver.
        self.tx_power_data = min(P_rts_datamcs, max_tx_power)
        self.mac.phy.tracker_power_data.add(self.tx_power_data)

        store_transmission_power(distance_to_receiver, self.tx_power_data)

        # Computing Power for CTS Packet
        # P_cts is MCS 0 at D_max + interference at the transmitter.
        # AT: 2/18/26 Adding external interference to rts data ack, previous code commented
        # P_cts = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=parameters.INTF_Registry[self.id])
        interference_at_self = parameters.INTF_Registry[self.id]
        if interference_at_self is None: interference_at_self = 0
        P_cts = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=interference_at_self+self.external_interference)
        max_tx_power_ctrl = parameters.getMaximumTransmitPower(parameters.CONTROL_MCS_INDEX)
        self.tx_power_cts = min(P_cts, max_tx_power_ctrl)
        self.mac.phy.tracker_power_cts.add(self.tx_power_cts)

        if parameters.PRINT_LOGS:
            print(f"Node: {self.env.now}: {self.name} RTS_default: {P_rts_default:.4f}, data_mcs: {data_mcs}, {distance_to_receiver=}, max_power: {max_tx_power:.4f}, RTS_datamcs: {P_rts_datamcs:.4f}, RTS: {self.tx_power_rts:.4f}, DATA: {self.tx_power_data:.4f}, CTS: {self.tx_power_cts:.4f}")
    
    
    # ANDRES: new function to increase/decrease the MCS index in our node in all the routes that is involved
    @profile
    def updateMCSindex(self, increase: bool) -> None:
        for eachRouteIdx in parameters.Route_Details:
            eachRoute = parameters.Route_Details[eachRouteIdx]['Route']
            # If invalid route, do not contend for the channel access and save energy
            if  (not parameters.Route_Details[eachRouteIdx]['ValidRoute']): break
            eachRoute_str = ["Node" + str(eachNode) for eachNode in eachRoute]; #print("node.py: route:",eachRoute_str)
            if (self.name in eachRoute_str):
                hopId = eachRoute_str.index(self.name)
                if parameters.Route_Details[eachRouteIdx]['MCS_Index'][hopId] < 7 and increase:
                    parameters.Route_Details[eachRouteIdx]['MCS_Index'][hopId] += 1
                if parameters.Route_Details[eachRouteIdx]['MCS_Index'][hopId] > 0 and not increase:
                    parameters.Route_Details[eachRouteIdx]['MCS_Index'][hopId] += -1
        return

    # Sh: Return next hop node name
    # ANDRES: modified the code so that now it also returns the previous node (if possible). // Now it also returns the route index
    @profile
    def findNextHop(self, flowSrc: str, flowDst: str) -> tuple[Optional[str], int, Optional[str], Optional[int]]:
        nexthopNode: Optional[str] = None
        prevhopNode: Optional[str] = None
        mcs_index: int = parameters.DEFAULT_DATA_MCS_INDEX
        eachRouteIdx: Optional[int] = None
        for eachRouteIdx in parameters.Route_Details:
            eachRoute = parameters.Route_Details[eachRouteIdx]['Route']
            if (("Node"+str(eachRoute[0]) == flowSrc) & ("Node"+str(eachRoute[-1]) == flowDst)):
                
                # If invalid route, do not contend for the channel access and save energy
                if  (not parameters.Route_Details[eachRouteIdx]['ValidRoute']): break

                eachRoute_str = ["Node" + str(eachNode) for eachNode in eachRoute]; #print("node.py: route:",eachRoute_str)
                route_len = len(eachRoute_str)
                if (self.name in eachRoute_str):
                    hopId = eachRoute_str.index(self.name)
                    if self.name != flowSrc:
                        prevhopNode = eachRoute_str[hopId-1]
                    if hopId == route_len - 1: break
                    nexthopNode = eachRoute_str[hopId+1]
                    mcs_index = parameters.Route_Details[eachRouteIdx]['MCS_Index'][hopId]
                    break
        return (nexthopNode, mcs_index, prevhopNode, eachRouteIdx)
    
    # MC: Generate packets at the source node and enqueue them
    @profile
    def PacketGeneration(self, idx: int):
        # Wait until packet gernation start time
        # yield self.env.timeout(parameters.PKT_GENERATION_START_TIME) # SD: Commented, starting PacketGeneration process after (PKT_GENERATION_START_TIME)s in pheromone.py

        print("node.py: Inside packet generation function at node: %s. Curr time: %f, Packet inter arrival time: %f" % (self.name, (self.env.now*1e-9), (parameters.PACKET_INTER_ARRIVAL_TIME[idx] + int(random.random() * 10000) - 5000)*1e-9))
        while (self.env.now <= parameters.PKT_GENERATION_END_TIME):
            yield self.env.timeout(parameters.PACKET_INTER_ARRIVAL_TIME[idx] + int(random.random() * 10000) - 5000)  # Randomness of [-5,5] micro seconds in packet generation
            #pkt_length = random.randint(0, parameters.MAX_MAC_PAYLOAD_LENGTH)
            
            route = parameters.Route_Details[idx]['Route']
            srcNode = parameters.NODE_REGISTRY[f'Node{route[0]}'].name
            dstNode = parameters.NODE_REGISTRY[f'Node{route[-1]}'].name
            
            pkt_length = parameters.MAX_MAC_PAYLOAD_LENGTH  # Sh: Use a fixed packet length

            # Create a new packet id to track packet in the queue. It is different from the id created at the time of packet transmission.
            pkt_id = str(round(self.env.now)) + '_' + srcNode + "_" + dstNode + "_" + parameters.NUMBERS_TO_WORDS[0]

            # Enqueue the packet
            self.EnqueuePacket(pkt_length, pkt_id, dstNode)
            self.num_packets_generated += 1
        print("node.py: Packet generation complete at node: %s." % (self.name))
        return
    
    # Sh: Functionalities for packet queuing, handles buffer overflow, and calls wrapper function to initiate packet sending.
    @profile
    def EnqueuePacket(self, pkt_length: int, pkt_id: str, flowDest: str):
		#print("Check 1: node:", self.name, "pkt id:", pkt_id)	
        
        # Make sure the packet isn't expired
        if (self.HasPacketExpired(pkt_id)[0]): # AT: Should only be checking one pkt here, not a combined packet
            # DONE: 10/23 Get interference values from table
            self_intf_reg = parameters.INTF_Registry[getNodeIdx(self.name)]
            _ = self.stats.loqQueueStats(self.name, pkt_id, 0, self.env.now, self.env.now, self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if self_intf_reg is None else self_intf_reg, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_EXPIRY, True)
        else:
            # MC: Add packet to the queue
            if not self.q.full():
                
            
                # Sh: Log packet generation
                #print("Check 2: node:", self.name, "pkt id:", pkt_id)
                # DONE: 10/23 Get interference values from table
                aNewPacket = self.stats.isPacketNew(self.name, pkt_id)
                if not aNewPacket:
                    # Do not enqueue this packet! It has already been included.
                    #print("Check 3: node:", self.name, "pkt id:", pkt_id)
                    return
                else:
                    self_intf_reg = parameters.INTF_Registry[getNodeIdx(self.name)]
                    self.stats.loqQueueStats(self.name, pkt_id, 0, self.env.now, float("inf"), self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if self_intf_reg is None else self_intf_reg, np.nan)
            
                with self.lock: # Sh
                    #print("Check 4: node:", self.name, "pkt id:", pkt_id)													  
                    if Use_Strict_FIFO_Order:
                        #print("Check 5: node:", self.name, "pkt id:", pkt_id)													  
                        self.q.put((0, flowDest, pkt_length, pkt_id))    # (Unused, Dest node, Packet Length, Packet Id)
                    else:
                        #print("Check 6: node:", self.name, "pkt id:", pkt_id)													  
                        self.q.put((1, flowDest, pkt_length, pkt_id))    # (Packet Priority, Dest node, Packet Length, Packet Id)
            
                if parameters.PRINT_LOGS:
                    print('node.py: Adding pkt: %s to node %s queue! Updated queue size = %d, %s, %s, %s' % (pkt_id, self.name, self.q.qsize(), f"{self.inKeepSending=}", f"{self.mac.hasPacket=}", f"{self.mac.hasACKToSend=}"))            

            else:   # Buffer overflow
                #print("Check 7: node:", self.name, "pkt id:", pkt_id)													  
                if parameters.PRINT_LOGS:
                    print('node.py: Node %s queue is full. Dropping this packet %s' % (self.name, pkt_id))
                # Sh: Log packet drop due to overflow
                # MC: Edit: Have to call function twice due to first having to register packet and then change the drop status
                # AT: Added write_new to create packets w/ a drop status without having to call twice
                self_intf_reg = parameters.INTF_Registry[getNodeIdx(self.name)]
                _ = self.stats.loqQueueStats(self.name, pkt_id, 0, self.env.now, self.env.now, self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if self_intf_reg is None else self_intf_reg, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_OVERFLOW, True)
                #print("Check 8: node:", self.name, "pkt id:", pkt_id)													  

                # Else case to delete packet at the bottom of the queue
                #_ = self.q.get(parameters.MAX_QUEUE_SIZE - 1)
        
        # Sh: Call wrapper function to send it.
        # print("Time:", self.env.now, "EnqueuePacket: Node:", self.name, "Node status:", self.mac.hasPacket, "Keep Sending:", self.inKeepSending, "pkt:", pkt_id)
        if ((not self.inKeepSending) and (not self.mac.hasPacket)):
			#print("Check 9: node:", self.name, "pkt id:", pkt_id)
            
            self.env.process(self.keepSending())
		#print("Check 10: node:", self.name, "pkt id:", pkt_id)													   
        return


    # Sh: Check packet expiry
    @profile
    def HasPacketExpired(self, pktId: str, useCaution: bool = True) -> list[bool]:
        # if '_COMBINED_' in pktId:
        #     print(f"node.py: HasPacketExpired: pktId: {pktId}, useCaution: {useCaution}")
        #     print(f"node.py: Considers first packet time - pktId.split('_')[0]: {int(pktId.split('_')[0])/(1e9)}")
        # While using caution, preemptively drop packets that are closer to expiration. Depends on packet transmission delay (for 2kB, ~6 ms), channel quality and est. time-to-destination.
        # print(f"node.py: HasPacketExpired, {self.env.now} - {int(pktId.split('_')[0])} -> {self.env.now - int(pktId.split('_')[0])} >= {parameters.PACKET_TTL - (10e-3 if useCaution else 0)}")
        pkts_to_check = pktId.split('_COMBINED_')
        return [self.env.now - int(each_pkt.split('_')[0]) >= parameters.PACKET_TTL - (10e-3 if useCaution else 0) for each_pkt in pkts_to_check]
        # return (self.env.now - int(pktId.split('_')[0]) >= parameters.PACKET_TTL - (10e-3 if useCaution else 0))


    # MC: Drop packets due to TTL expiry. # TODO: Not tested for strict priority queue
    @profile
    def DropPacketDueToExpiry(self, queue: Queue[qEntry]):
        interval = min(min(parameters.PACKET_INTER_ARRIVAL_TIME)*parameters.STEP_SIZE, parameters.PACKET_TTL) # Used step size to reduce frequent recomputations
        while True:
            yield self.env.timeout(interval)
            # Iterates through all packets in queue once
            with self.lock: # Sh
                curr_q_size = queue.qsize()
                for _ in range(curr_q_size):
                    curr_pkt = queue.get()
                    # Determine whether to drop the packet or not
																							 
                    if (self.HasPacketExpired(curr_pkt[3])[0]): # AT: Should only be checking one pkt here, not a combined packet
                        self.packetsExpired += 1
                        # Record packet drop
                        if parameters.PRINT_LOGS:
                            print('node.py: Time %d: Packet %s expired at node %s. Drop from queue!\n' % (self.env.now, curr_pkt[3], self.name))
                        self_intf_reg = parameters.INTF_Registry[getNodeIdx(self.name)]
                        _ = self.stats.loqQueueStats(self.name, curr_pkt[3], 0, float(curr_pkt[3].split('_')[0]), self.env.now, self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if self_intf_reg is None else self_intf_reg, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_EXPIRY)

                        '''if self.packetsExpired > 3:
                            self.updateMCSindex(True)
                            self.txPower_increase = True
                            parameters.NODE_REGISTRY[self.name] = self'''

                    else: queue.put(curr_pkt)

    
    # Sh: Periodic two-setp queue management. Step 1: Reshuffle queue based on survivability (=TTE/ETD) score, Step 2: Discard packet that has survivability score < Threshold.
    # If flow priroity is considered, there is another step: reshuffle the queue based on survivability score/priority.
    @profile
    def QueueManagement(self, queue: Queue[qEntry] | PriorityQueue[qEntry]):
        nodeId = getNodeIdx(self.name)
        # Initial wait
        yield self.env.timeout(parameters.LOCATION_UPDATE_INTERVAL/parameters.STEP_SIZE)
        while True:
            yield self.env.timeout(parameters.QUEUE_UPDATE_INTERVAL)
            with self.lock:
                curr_time = self.env.now/1e9  # in seconds
                curr_q_size = queue.qsize()
                # Skip computation if this node has an empty queue
                if curr_q_size == 0: continue

                if parameters.PRINT_LOGS: 
                    print("In node.py: Before queue management at node:", self.name, "at time:",curr_time, "Queue size:",curr_q_size)

                # Store the packets in a temp queue
                tmp_q: list[qEntry] = []
                for _ in range(curr_q_size):
                    tmp_q.append(queue.get())

                # Get ETD from current node to destination and flow priroity
                ETD_to_dest: list[ETD_to_destEntry] = []    # Stores flow id, remaining route to destination from this node, remaining ETD
                for routeNo in parameters.Route_Details:
                    route = parameters.Route_Details[routeNo]['Route']
                    if (nodeId in route):
                        node_idx = route.index(nodeId)
                        route_src = route[0]; route_dest = route[-1]
                        rem_route = route[node_idx:]    # Remaining route to destination from this node
                        rem_ETD = 0.0                   # ETD to destination from this node
                        for hopNo in range(len(rem_route)-1):  # Do not consider destination node in ETD calculation  
                            rem_ETD += float(parameters.Node_Stats[rem_route[hopNo]]['PST']) # No moving average here
                        flow_priority = parameters.Route_Details[routeNo]['Priority']
                        ETD_to_dest.append({'Flow Id':f"Node{route_src}_Node{route_dest}",'Remaining Route':rem_route,'Remaining ETD':rem_ETD,'Priority':flow_priority})

                # Compute packet survivability score
                @profile
                def get_sur_score(pkt: qEntry, pri: bool = False) -> tuple[float, float]:
                    # Get flow id (source and destination pair)
                    parts = pkt[3].split('_'); flowID = '_'.join(parts[1:3]); #print("flow id",flowID)

                    # Initialization: Default is UDP. When no route for the flow exists at this node, wait for topology update and hope to get a route. 
                    # This requires moving the packet back in the queue. If TCP is used, we should not have come here as the packets should have been 
                    # already reinserted to the source node during the last topology update.
                    # Otherwise, if the route exists, get the remaining ETD.
                    rem_ETD = float("inf"); flow_priority = 1
                    for eachFlow in ETD_to_dest:
                        if (eachFlow['Flow Id'] == flowID):
                            rem_ETD = eachFlow['Remaining ETD']
                            if pri or not parameters.QUEUE_MANAGEMENT_PLUS: flow_priority = eachFlow['Priority']
                            else: flow_priority = 1
                            break
                    sur_score = ((parameters.PACKET_TTL / 1e9) - self.getPktAge(pkt[3],curr_time)) / max(rem_ETD,1e-9)    # Avoids devide by 0 error
                    if parameters.PRINT_LOGS:
                        print(f"node.py: At {self.name} : {self.getPktAge(pkt[3], curr_time)} / {rem_ETD = }; {sur_score = }; {flowID = }")
                    return (sur_score, sur_score/flow_priority)

                pkts_and_sur_scores = [(packet, *get_sur_score(packet)) for packet in tmp_q]

                # Step 1: Rearrange the queue based on the survivability score
                pkts_and_sur_scores.sort(key=lambda item: item[1])
                
                # Step 2: Discard packets (only from first X packets) that have survivability score < Threshold
                pkts_to_keep: list[tuple[qEntry, float, float]] = []
                for pkt_counter, item in enumerate(pkts_and_sur_scores):
                    #print("Checking pkt:", item[0], "pkt counter:", pkt_counter, "sur score:", item[1], "threshold:", parameters.SUR_THRESHOLD)
                    if ((pkt_counter >= parameters.FIRST_N_PKTS) or (item[1] >= parameters.SUR_THRESHOLD)):
                        pkts_to_keep.append(item)
                    else:
                        # Record packet drop
                        if parameters.PRINT_LOGS:
                            print('node.py: Time %d: Packet %s survivability score is %f (< %0.2f) at node %s. Drop from queue! Packet counter: %d\n' % (self.env.now, item[0][2], item[1], parameters.SUR_THRESHOLD, self.name, pkt_counter))
                        self_intf_reg = parameters.INTF_Registry[getNodeIdx(self.name)]
                        _ = self.stats.loqQueueStats(self.name, item[0][3], 0, float(item[0][3].split('_')[0]), self.env.now, self.mac.freeze_counter, self.mac.phy.received_power_current_timeslot if self_intf_reg is None else self_intf_reg, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_Q_MNGMNT)

                # Additional step: Rearrange the queue based on survivability score and priority
                if parameters.QUEUE_MANAGEMENT_PLUS:
                    pkts_to_keep = [(packet[0], *get_sur_score(packet[0], True)) for packet in pkts_to_keep]
                    if parameters.USE_FLOW_PRIORITY:
                        pkts_to_keep.sort(key=lambda item: (item[1], item[2]))
                elif parameters.USE_FLOW_PRIORITY: 
                    pkts_to_keep.sort(key=lambda item: item[2])    
                
                # Reinsert the remaining packets
                for each_remain_pkt, _, _ in pkts_to_keep:
                    queue.put(each_remain_pkt)

                if parameters.PRINT_LOGS:
                    print("In node.py: After queue management at node:", self.name, "at time:",curr_time, "Queue size:",queue.qsize())

    # Sh: Get packet age (in seconds)
    @profile
    def getPktAge(self, pktID: str, curr_time: float) -> float:
        return curr_time - int(pktID.split('_')[0])/(1e9)

    # Sh: Get TTE of head-of-the-line packet
    @profile
    def getHOLPktTTE(self, curr_time: float) -> float:
        with self.lock:
            if not self.q.empty():
                HOL_pkt = self.q.queue[0]
                pktTTE = parameters.PACKET_TTL - self.getPktAge(HOL_pkt[3],curr_time); #print("HOL pkt:",HOL_pkt,"TTE:",pktTTE)
            else:
                # If there is no packet in the queue, use max time allowed for a packet to be in the queue.
                pktTTE = parameters.PACKET_TTL
        return pktTTE
