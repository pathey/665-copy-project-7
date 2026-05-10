import copy
from typing import Callable, Dict, Optional, ParamSpec, TypeVar, TYPE_CHECKING

import numpy as np
import simpy
import random

import phy
from macPacket import MacPacket
import parameters
from simpy.events import ProcessGenerator
from stats import Stats

# Sh
from RouteFunctions import getFlowSrc, getFlowDest, getRoute, getNodeIdx
from ether import Ether, computeDistance

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

class Mac(object):
    @profile
    def __init__(self, node: 'Node'):
        self.node: 'Node' = node
        self.env: simpy.Environment = self.node.env
        self.name: str = self.node.name
        self.id: int = getNodeIdx(self.name)
        self.ether: Ether = self.node.ether
        self.latitude: float = self.node.latitude
        self.longitude: float = self.node.longitude
        self.stats: Stats = self.node.stats

        # ANDRES 12/03/25: Added new attributes to mac layer since they are going to be needed in the phy layer.
        # AT: 2/16/26: Access powers at node level. Having a primitive that must be updated at all layers is prone to error.
        # self.tx_power_data: float = self.node.tx_power_data
        # self.tx_power_ack: float = self.node.tx_power_ack
        # self.tx_power_rts: float = self.node.tx_power_rts
        # self.tx_power_cts: float = self.node.tx_power_cts
        
        self.external_interference: float = self.node.external_interference
        self.routing_power: float = self.node.routing_power

        self.phy: phy.Phy = phy.Phy(self)
        self.pendingPackets: Dict[str, simpy.Process] = {}    # associate packets I'm trying to transmit and timeouts for retransmissions
        self.retransmissionCounter: Dict[str, int] = {}     # associate packets I'm trying to transmit and transmission attempts
        self.isSensing: bool = False
        self.packetsToSend = []
        self.sensing: Optional[simpy.Process] = None  # keep sensing process
        self.hasPacket: bool = False  # Sh: Track node state. If true, node already has a data packet in hand.
        self.hasACKToSend: bool = False # AT: To prevent extracting a new packet from queue until ACK is sent for received data packet
        
        self.freeze_backoff_until: float = 0
        self.unfreeze_add_difs_consumed: bool = True
        
        self.frozen_this_loop: bool = False
        self.freeze_counter: int = 0
        self.unfrozen_this_loop: bool = False
        self.unfreeze_counter: int = 0
        
        self.cts_retry: int = 0
        self.ack_retry: int = 0
        self.sent_ack: int = 0
        self.send_ack_per_packet: int = 0
        self.num_retransmissions: int = 0
        self.num_pkts_retransmitted: int = 0
        self.num_received: int = 0
        self.num_packets_received: int = 0
        self.num_forwards: int = 0
        self.num_packets_forwarded: int = 0 # AT: Pkt is counted as forwarded when corresponding ACK is received
        self.num_packets_serviced: int = 0 # AT: Pkt is counted as serviced the first time it is sent
        self.received_packets: dict[str, float] = {}

    @profile
    def send(self, destination: str, payloadLength: int, id: str, nPktsAgg: int, route_index: int, mcs_index: Optional[int] = None) -> ProcessGenerator:    # Sh: Added mcs index
        mcs_index = parameters.DEFAULT_DATA_MCS_INDEX if mcs_index is None else mcs_index
        
        self.hasPacket = True # Sh: Update node state
        # print(f"[MAC], SEND: in, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        if parameters.PRINT_LOGS: print("In send mac: node:", self.name, "with pkt", id)
        
        length = payloadLength + parameters.MAC_HEADER_LENGTH
        # ANDRES: HARD FIX
        # node_id = getNodeIdx(self.name)
        macPkt = MacPacket(self.name, destination, int(length), id, False, self.node.tx_power_data, self.phy.received_power_current_timeslot, self.routing_power, mcs_index, route_index, [False]*nPktsAgg)
        
        # # SD: ???
        # macPkt = macPacket.MacPacket(self.name, destination, length, id, False, self.transmission_power, self.routing_power, mcs_index, route_index)

        # Sh: Check if it is the flow source and transmtting for its own flow destination node
        pktOriginatedByFlowSrc = False
        flowSrc = getFlowSrc(id); flowDst = getFlowDest(id)
        if ((self.name == flowSrc) and (getRoute(flowSrc,flowDst) != [])):
            pktOriginatedByFlowSrc = True

        
        self.retransmissionCounter[macPkt.id] = 0
        # sensing phase
        if self.phy.isSending:   # I cannot sense while sending
            if self.phy.transmission is not None:
                if parameters.PRINT_LOGS: print(f"TRANSMISSION: {self.name}, {self.env.now}, {self.phy.transmission}")
                yield self.phy.transmission   # wait for my phy to finish sending other packets


        if self.isSensing: # I'm sensing for another packet, I wait
            if self.sensing is not None:
                if parameters.PRINT_LOGS: print(f"SENSING: {self.name}, {self.env.now}, {self.sensing}")
                yield self.sensing

        
        self.sensing = self.env.process(self.waitIdleAndSend(macPkt))
        yield self.sensing
        self.stats.logTransmittedPacket(id, self.env.now, length, macPkt.MCS_index, macPkt.tx_power, pktOriginatedByFlowSrc) # Sh: Added more fields // ANDRES: added the field for tx_power for the results in the .csv
        self.num_packets_serviced += len(macPkt.aggregated_ack_status)
        # print(f"[MAC], SEND: out, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
    
    @profile
    def handleReceivedPacket(self, macPkt: MacPacket, sinr: list[float], pkts_received: list[bool]):   # Sh: Added sinr argument
        # print(f"[MAC], handleReceivedPacket: in, @{self.env.now}, {self.name}, {self.hasPacket = },|{macPkt.id},{macPkt.ack=},{macPkt.aggregated_ack_status}|, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        assert len(sinr) == len(pkts_received), "sinr contains the values, pkts_received contains the flags, so they should be the same length. sinr should only be used for stats.logDeliveredPacket"

        if macPkt.destination == self.name and not macPkt.ack:  # send ack to normal packets

            # Check if this is a combined packet
            pkt_ids = macPkt.id.split('_COMBINED_')
            is_combined = len(pkt_ids) > 1
            num_packets = len(pkt_ids)
            
            if parameters.PRINT_LOGS:
                print(f"Combined Id: {macPkt.id} | Individual Ids: {pkt_ids}")
            
            for i, pkt_id in enumerate(pkt_ids):
                if not pkts_received[i]: continue
                
                # Sh: Obtain flow source and destination nodes
                flowSrc = getFlowSrc(pkt_id)
                flowDst = getFlowDest(pkt_id)

                if parameters.PRINT_LOGS:
                    print('mac.py: Time %d: %s MAC receives packet %s from %s. Flow src = %s, dst = %s. and sends ACK' % (self.env.now, self.name, pkt_id, macPkt.source, flowSrc, flowDst))
                self.node.receive(pkt_id, macPkt.source)
                self.stats.logDeliveredPacket(pkt_id, self.env.now, self.name == flowDst, sinr[i])    # Sh: Added sinr value and a flag to indicate if it is the flow destination.
                
                # Sh: If this node is not the flow destination, forward the received packet to next hop node
                if (self.name != flowDst):
                    _, flowBasedID = self.stats.packetIDs(pkt_id)
                    # For combined packets, split the length equally (since we don't track individual lengths)
                    pkt_length = (macPkt.length-parameters.MAC_HEADER_LENGTH) // (num_packets if is_combined else 1)
                    
                    # Enqueue the packet for forwarding. Remove MAC header added (only in data packet) in Send() above
                    self.hasACKToSend = True
                    self.node.EnqueuePacket(pkt_length, flowBasedID, flowDst)
                else:
                    if parameters.PRINT_LOGS:
                        print("mac.py: Flow destination %s received packet from flow source: %s via last hop: %s\n" % (self.name, flowSrc, macPkt.source))

                if pkt_id in self.received_packets:
                    if parameters.PRINT_LOGS:
                        print(f"DUPLICATE RECIEVED PACKET: {pkt_id} 1st at {self.received_packets[pkt_id]} and now again at {self.env.now}")
                else:
                    self.received_packets[pkt_id] = self.env.now
                
                self.num_packets_received += 1
            self.num_received += 1
            # Send single ACK for the combined packet # This node can also maintain a table with  min (transmission power it computed in step 2, notified by rts)
            # AT: We should compute the power at the same index we are sending at, talk to Prof
            # P_ack is MCS 0 at D_max + interference at the transmitter.

            # AT: 2/18/26 Adding external interference to rts data ack, previous code commented
            # P_ack = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=parameters.INTF_Registry[getNodeIdx(macPkt.source)])
            intf = parameters.INTF_Registry[getNodeIdx(macPkt.source)]
            if intf is None: intf = 0
            P_ack = parameters.getTransmitPower(parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, interference=intf+parameters.NODE_REGISTRY[macPkt.source].external_interference)
            max_tx_power = parameters.getMaximumTransmitPower(parameters.CONTROL_MCS_INDEX)

            self.node.tx_power_ack = min(P_ack, max_tx_power)
            self.phy.tracker_power_ack.add(self.node.tx_power_ack)
            
            macPkt.aggregated_ack_status = pkts_received
            ack = MacPacket(self.name, macPkt.source, parameters.ACK_LENGTH, macPkt.id, True, P_ack, self.phy.received_power_last_timeslot, self.routing_power, parameters.CONTROL_MCS_INDEX, macPkt.route_index, copy.deepcopy(macPkt.aggregated_ack_status))   # Added new fields
            
            # If intended transmitter distance is less than some threshold, wait additional SIFS (2x) (for even lower distances it might be more)
            yield self.env.timeout(parameters.SIFS_DURATION * (macPkt.MCS_index+1)) # INFO: Tested for MCS 0 and 1
            
            if parameters.PRINT_LOGS:
                print(f"mac.py: {self.env.now} {self.name} sends ACK for {ack.id} to {ack.destination}")
            
            self.phy.send(ack)
            self.sent_ack += 1
            self.send_ack_per_packet += len(pkt_ids)
            self.hasACKToSend = False

        elif macPkt.destination == self.name:
            if parameters.PRINT_LOGS:
                print('mac.py: Time %d: %s MAC receives ACK %s from %s' % (self.env.now, self.name, macPkt.id, macPkt.source))

            # AT: If part of the combined packet was not received, we will resend it
            if all(macPkt.aggregated_ack_status):# and all(pkts_received): # AT: Since ACKs currently are not split up even if they represent a combined packet, we don't need to check if all parts of ACK packet was received
                self.hasPacket = False # Sh: Update node's state
            # print("In handlereceivepacket mac: node:", self.name, "node status:", self.hasPacket)

            if macPkt.id in self.pendingPackets:    # packet could not be in pendingPackets if timeout has expired but ack still arrive
                self.pendingPackets[macPkt.id].interrupt()
            else:
                if parameters.PRINT_LOGS:
                    print(f"mac.py: {self.env.now}, {self.name}, {macPkt.id} not in pendingPackets")

        # Sh: Delete NAV entry for this packet id from neighbor's NAV table.
        elif macPkt.ack:
            parameters.delNAVEntry(macPkt.id, getNodeIdx(self.name))
            #if parameters.PRINT_LOGS: print(f"After deleting NAV entry due to ACK reception at node: {self.name}, NAV Table: {parameters.NAV_Table}\n")

        # print(f"[MAC], handleReceivedPacket: out, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")

    @profile
    def _partition_pkts(self, pkt_ids: list[str], flags: list[bool]) -> tuple[list[str], list[str]]:
        assert len(pkt_ids) == len(flags), f"{len(pkt_ids) = }, {len(flags) = }. The number of pkt_ids and flags to check against must be the same."
        
        successful_pkts: list[str] = []
        unsuccessful_pkts: list[str] = []
        for each_pkt, success in zip(pkt_ids, flags, strict=True):
            if success:
                successful_pkts.append(each_pkt)
            else:
                unsuccessful_pkts.append(each_pkt)

        return successful_pkts, unsuccessful_pkts
    
    @profile
    def split_by_ack_status(self, pkt_ids: list[str], ack_status: list[bool]) -> tuple[list[str], list[str]]:
        return self._partition_pkts(pkt_ids, ack_status)
    
    @profile
    def split_by_expiry(self, pkt_ids: list[str], flowBasedID: str) -> tuple[list[str], list[str]]:
        expired_flags = self.node.HasPacketExpired(flowBasedID, False)
        return self._partition_pkts(pkt_ids, expired_flags)
    
    @profile
    def log_dropped_pkts(self, pkts: list[str], retx_cnt: int, reason: Optional[str], dst: str):
        if not pkts:
            return

        intf = parameters.INTF_Registry[getNodeIdx(self.name)]
        for each_pkt in pkts:
            hopID, flowID = self.stats.packetIDs(each_pkt)
            self.stats.loqQueueStats(self.name, flowID, retx_cnt, float(hopID.split('_')[0]),
                                     self.env.now, self.freeze_counter, 0 if intf is None else intf,
                                     np.nan, parameters.PKT_DROPPED, reason)
            if parameters.PRINT_LOGS:
                if reason == parameters.DROP_REASON_RETRYLIMIT:
                    print('mac.py: Time %d: %s reached max MAC retry limit for packet %s to %s. Discard!\n' % (self.env.now, self.name, each_pkt, dst))
                elif reason == parameters.DROP_REASON_EXPIRY:
                    print('mac.py: Time %d: Packet %s expired at node %s. Do not transmit anymore!\n' % (self.env.now, each_pkt, self.name))

    @profile
    def build_new_macPkt(self, old_pkt: MacPacket, pkt_ids: list[str]) -> MacPacket:
        per_pkt_payload = (old_pkt.length - parameters.MAC_HEADER_LENGTH) // len(old_pkt.id.split('_COMBINED_'))

        new_length = (per_pkt_payload * len(pkt_ids)) + parameters.MAC_HEADER_LENGTH

        return MacPacket(self.name, old_pkt.destination, int(new_length), '_COMBINED_'.join(pkt_ids), old_pkt.ack, 
                         self.node.tx_power_data, self.phy.received_power_current_timeslot, self.routing_power, old_pkt.MCS_index,
                         old_pkt.route_index, [False] * len(pkt_ids))

    def waitAck(self, macPkt: MacPacket, cts_timeout: bool) -> ProcessGenerator:
        # print(f"[MAC], waitACK: in, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        
        if parameters.PRINT_LOGS:
            print("mac.py: wait ack at node:", self.name, "for pkt:", macPkt.id, "timeout:", macPkt.NAV, "until:", self.env.now+macPkt.NAV)
        try:
            #yield self.env.timeout(parameters.ACK_TIMEOUT)
            yield self.env.timeout(macPkt.NAV)  # Sh
            # timeout expired, resend

            # Sh: Delete NAV entry for this packet id from neighbor's NAV table. If a new retransmission occurs, a new NAV entry will be created!
            parameters.delNAVEntry(macPkt.id)
            #if parameters.PRINT_LOGS: print("After deleting NAV entry from all nodes due to packet retransmission: NAV Table:",parameters.NAV_Table,"\n")

            pkt_ids = macPkt.id.split('_COMBINED_')

            self.pendingPackets.pop(macPkt.id)
            retx_cnt = self.retransmissionCounter.pop(macPkt.id)

            # Sh: Discard packets if retry limit is reached or TTL expired!
            _, flowBasedID = self.stats.packetIDs(macPkt.id)
            expired, alive = self.split_by_expiry(pkt_ids, flowBasedID)

            self.log_dropped_pkts(expired, retx_cnt, parameters.DROP_REASON_EXPIRY, macPkt.destination)

            if not alive:
                self.hasPacket = False
                # print(f"[MAC], waitACK: out alive timeout, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
                return
            
            if retx_cnt >= parameters.MAX_RETRY_LIMIT:
                self.log_dropped_pkts(alive, retx_cnt, parameters.DROP_REASON_RETRYLIMIT, macPkt.destination)
                self.hasPacket = False
                # print(f"[MAC], waitACK: out retx timeout, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
                return
            
            if len(alive) == len(pkt_ids): # AT: We don't have to create a new macpkt if there were no drops
                new_macPkt = macPkt
            else:
                new_macPkt = self.build_new_macPkt(macPkt, alive)
            
            self.retransmissionCounter[new_macPkt.id] = retx_cnt + 1
            if parameters.PRINT_LOGS:
                print('mac.py: Time %d: %s MAC retransmit %s (Retry No.: %d) to %s' % (self.env.now, self.name, new_macPkt.id, self.retransmissionCounter[new_macPkt.id], new_macPkt.destination))
            
            # sensing phase
            if self.phy.isSending:   # I cannot sense while sending
                if self.phy.transmission is not None:
                    # print(f"[MAC], waitACK: timeout yield phy, @{self.env.now}, {self.name}, {macPkt.id}, {self.phy.transmission.name} {self.phy.transmission.target}")
                    yield self.phy.transmission   # wait for my phy to finish sending other packets

            if self.isSensing: # I'm sensing for another packet, I wait
                if self.sensing is not None:
                    # print(f"[MAC], waitACK: yield sense, @{self.env.now}, {self.name}, {macPkt.id}, {self.sensing.name} {self.sensing.target}")
                    yield self.sensing

            for each_retx_pkt in new_macPkt.id.split('_COMBINED_'):
                self.stats.logRetransmission(self.env.now, each_retx_pkt)    # Sh: Added new field
                self.num_pkts_retransmitted += 1
            self.num_retransmissions += 1
            if cts_timeout:
                self.cts_retry += 1
            else:
                self.ack_retry += 1
            self.sensing = self.env.process(self.waitIdleAndSend(new_macPkt))
            # print(f"[MAC], waitACK: out, @{self.env.now}, {self.name}, {self.hasPacket = },|no ack|, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        except simpy.Interrupt:
            # ack received
            pkt_ids = macPkt.id.split('_COMBINED_')
            retx_cnt = self.retransmissionCounter.pop(macPkt.id)
            self.pendingPackets.pop(macPkt.id)

            successful_pkts, unsuccessful_pkts = self.split_by_ack_status(pkt_ids, macPkt.aggregated_ack_status)
            
            if parameters.PRINT_LOGS:
                print("SINRACK", self.name, "id: ", macPkt.id, " success:", successful_pkts, "fail:", unsuccessful_pkts)
                print(macPkt.aggregated_ack_status)

            # Handle successful packets
            if successful_pkts:
                self.num_forwards += 1
                self.num_packets_forwarded += len(successful_pkts)
            
                pkt_id_only_success = '_COMBINED_'.join(successful_pkts)
                # Sh: Log packet dequeue time after successful forwarding
                _, success_flowBasedID = self.stats.packetIDs(pkt_id_only_success)
                # DONE: 10/23 Use table to get the interference value for both tx and rx
                reg_self_intf = parameters.INTF_Registry[getNodeIdx(self.name)]
                reg_dest_intf = parameters.INTF_Registry[getNodeIdx(macPkt.destination)]
                if reg_self_intf is None: reg_self_intf = self.phy.received_power_current_timeslot
                if reg_dest_intf is None: reg_dest_intf = np.nan
                
                self.stats.logSuccessfulForward(pkt_id_only_success, self.env.now)
                self.stats.loqQueueStats(self.name, success_flowBasedID, retx_cnt, self.env.now, self.env.now, self.freeze_counter, reg_self_intf, reg_dest_intf, parameters.PKT_FORWARDED_SUCCESSFULLY)
                if parameters.PRINT_LOGS:
                    print('mac.py: Time %d: %s ackownledges received ACK for packet %s to %s. (%d/%d) were successful. Dequeued from MAC queue.\n' % (self.env.now, self.name, pkt_id_only_success, macPkt.destination, len(successful_pkts), len(pkt_ids)))
            
            # Handle unsuccessful packet
            if unsuccessful_pkts:
                _, flowBasedID = self.stats.packetIDs('_COMBINED_'.join(unsuccessful_pkts))
                expired, alive = self.split_by_expiry(unsuccessful_pkts, flowBasedID)

                self.log_dropped_pkts(expired, retx_cnt, parameters.DROP_REASON_EXPIRY, macPkt.destination)

                if not alive:
                    # print(f"[MAC], waitACK: out alive ack unsuc, @{self.env.now}, {self.name}, {self.hasPacket = },|{successful_pkts=},{unsuccessful_pkts=}|, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
                    self.hasPacket = False
                    return

                if retx_cnt >= parameters.MAX_RETRY_LIMIT:
                    self.log_dropped_pkts(alive, retx_cnt, parameters.DROP_REASON_RETRYLIMIT, macPkt.destination)
                    self.hasPacket = False
                    # print(f"[MAC], waitACK: out retx ack unsuc, @{self.env.now}, {self.name}, {self.hasPacket = },|{successful_pkts=},{unsuccessful_pkts=}|, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
                    return
                
                if len(alive) == len(pkt_ids): # AT: We don't have to create a new macpkt if there were no drops
                    new_macPkt = macPkt
                else:
                    new_macPkt = self.build_new_macPkt(macPkt, alive)
                
                self.retransmissionCounter[new_macPkt.id] = retx_cnt + 1
                if parameters.PRINT_LOGS:
                    print('mac.py: Time %d: %s MAC retransmit %s (Retry No.: %d) to %s' % (self.env.now, self.name, new_macPkt.id, self.retransmissionCounter[new_macPkt.id], new_macPkt.destination))
                
                # sensing phase
                if self.phy.isSending:   # I cannot sense while sending
                    if self.phy.transmission is not None:
                        # print(f"[MAC], waitACK: ack yield phy, @{self.env.now}, {self.name}, {macPkt.id}, {self.phy.transmission.name} {self.phy.transmission.target}")
                        yield self.phy.transmission   # wait for my phy to finish sending other packets

                if self.isSensing: # I'm sensing for another packet, I wait
                    if self.sensing is not None:
                        # print(f"[MAC], waitACK: ack yield sense, @{self.env.now}, {self.name}, {macPkt.id}, {self.sensing.name} {self.sensing.target}")
                        yield self.sensing

                for each_retx_pkt in new_macPkt.id.split('_COMBINED_'):
                    self.stats.logRetransmission(self.env.now, each_retx_pkt)    # Sh: Added new field
                    self.num_pkts_retransmitted += 1
                self.num_retransmissions += 1
                if cts_timeout:
                    self.cts_retry += 1
                else:
                    self.ack_retry += 1
                
                self.sensing = self.env.process(self.waitIdleAndSend(new_macPkt))
            # print(f"[MAC], waitACK: out, @{self.env.now}, {self.name}, {self.hasPacket = },|{successful_pkts=},{unsuccessful_pkts=}|, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")

    # Sh: Add NAV entries for 1-hop neighbors (found based on distance required to decode packets transmitted using parameters.CONTROL_MCS_INDEX index)
    @profile
    def addNAVEntryAtAllOneHopNghbrs(self, txId: int, rxId: int, pkt: MacPacket, duration: float, cutoffTime: float, RTS_flag: bool = False) -> None:
        # print(f"[MAC], addNAVEntryAtAllOneHopNghbrs: in, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        packetID = pkt.id
        # Get coordinates
        powerNode = parameters.NODE_REGISTRY[f"Node{txId}"]
        latitude = powerNode.latitude
        longitude = powerNode.longitude
        
        if (getNodeIdx(self.name) == txId):  # Transmitter node
            latitude = self.latitude
            longitude = self.longitude
            powerNode = self.node
        else:   # Receiver node
            # Find its coordinates
            for _, thisNode in self.ether.channelsAndListeningNodes:
                if (getNodeIdx(thisNode.name) == txId):
                    latitude = thisNode.latitude
                    longitude = thisNode.longitude
                    powerNode = thisNode
                    break
        
        if parameters.PRINT_LOGS:
            print(f"mac addNav with {packetID}, {txId=} {rxId=}, found: {powerNode.name}")
        
        for _, eachNode in self.ether.channelsAndListeningNodes:
            eachNodeId = getNodeIdx(eachNode.name)
            # Skip self and transmitter/receiver node
            if ((eachNodeId == txId) or (eachNodeId == rxId)): 
                continue
            distance = computeDistance(latitude, longitude, eachNode.latitude, eachNode.longitude)
            # ANDRES: now uses the distance that comes from each node's routing power and the correspondent external interference at the destination.
            # SD: modified to make RTS using txNode.transmission_power
            if RTS_flag:
                # Use power for RTS
                txNode_power = self.node.tx_power_rts
            else:
                # Compute power used for CTS
                # min_transmit_power = parameters.getTransmitPower(distance=parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, noise=parameters.NOISE_FLOOR, interference=pkt.intf_power) # The interference 
                # max_transmit_power = parameters.getMaximumTransmitPower(parameters.CONTROL_MCS_INDEX)
                txNode_power = self.node.tx_power_cts
                
            # txNode_power = pkt.tx_power if RTS_flag else self.node.routing_power # SD: ??? CTS uses to default routing power, RTS uses to node's transmission power
            # txNode_power = parameters.getTransmitPower(distance=parameters.MAXIMUM_ROUTING_RANGE, mcs_index=parameters.CONTROL_MCS_INDEX, noise=parameters.NOISE_FLOOR, interference=powerNode.mac.phy.received_power_current_timeslot)
            
            # rangeMax = parameters.txRangeAtInterferenceLevel(txNode_power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR'], powerNode.external_interference)
            rangeMax = parameters.txRangeAtInterferenceLevel(txNode_power, parameters.MCS_SNR_TABLE[parameters.CONTROL_MCS_INDEX]['MinSNR']) # Changed 2/12/26
            if (distance <= rangeMax): # Range at which RTS/CTS can be decoded using parameters.CONTROL_MCS_INDEX index
                # Do not create a duplicate
                if (not (packetID in parameters.NAV_Table[eachNodeId])):
                    # Precaution to reduce memory usage at the cost of more computations
                    parameters.deleteOldNAVEntriesAtNode(eachNodeId, cutoffTime)
                    # print(f"{self.env.now} Adding NAV entry for packet id: {packetID}, duration: {duration} at node id {eachNodeId}.")
                    parameters.NAV_Table[eachNodeId][packetID] = duration
                    if duration < parameters.Min_NAV_Expiry[eachNodeId]: parameters.Min_NAV_Expiry[eachNodeId] = duration
            # if parameters.PRINT_LOGS:
            #     print(f"mac.py: NAV: {self.env.now} dist from {powerNode.name} to {eachNodeId} = {distance} <= {rangeMax} = {distance <= rangeMax}")
                # print(parameters.NAV_Table)
        # print(f"[MAC], addNAVEntryAtAllOneHopNghbrs: out, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        return

    # @profile
    def waitIdleAndSend(self, macPkt: MacPacket) -> ProcessGenerator:
        self.isSensing = True
        # print(f"[MAC], waitIdleAndSend: in, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
        timeout = parameters.DIFS_DURATION
        backoff = 0
        
        if not parameters.CSMA_CA_BACKOFF:
            backoff = random.randint(0, min(int(pow(2,self.retransmissionCounter[macPkt.id])*parameters.CW_MIN), parameters.CW_MAX)-1) * parameters.SLOT_DURATION
        elif parameters.CSMA_CA_BACKOFF:
            flow_priority = 1
            for eachRouteIdx in parameters.Route_Details:
                if (macPkt.id.split('_')[-3] == 'Node' + str(parameters.Route_Details[eachRouteIdx]['Route'][0])) and (macPkt.id.split('_')[-2] == 'Node' + str(parameters.Route_Details[eachRouteIdx]['Route'][-1])):
                    flow_priority = parameters.Route_Details[eachRouteIdx]['Priority']
                    break
            if flow_priority >= parameters.HIGH_PRIORITY:
                backoff = random.randint(0, min(int(pow(2,self.retransmissionCounter[macPkt.id])*parameters.CW_MIN/flow_priority), parameters.CW_MAX)-1) * parameters.SLOT_DURATION
                #backoff = random.randint(0, (parameters.CW_MIN / 2) + self.retransmissionCounter[macPkt.id] -1) * parameters.SLOT_DURATION
            else:
                backoff = random.randint(0, min(pow(2,self.retransmissionCounter[macPkt.id])*parameters.CW_MIN, parameters.CW_MAX)-1) * parameters.SLOT_DURATION
        
        # Print backoff here the first time it is set when the node transmits a packet
        if parameters.PRINT_LOGS:
            print(f"mac.py: {self.env.now}, {self.name}, {backoff = }, {macPkt.id}")
        
        timeout += backoff

        # Sh: Get node ids of transmitter and receiver nodes
        txID = getNodeIdx(macPkt.source)
        rxID = getNodeIdx(macPkt.destination)

        while True:
            try:
                # Sh: Wait until the active NAV expires
                currTime = round(self.env.now) # in ns
                maxNav = parameters.getMaxNAV(txID, currTime)
                # print(f"mac.py: nav: Getting {txID}'s maxNAV: {maxNav} ({maxNav - currTime})")
                yield self.env.timeout(max(0, maxNav - currTime))

                while timeout > 0:
                    addDIFSDuration = False # Set it if waited for NAV
                    while True:
                        # Recheck NAV after each slot because (1) a node might receive another packet with a higehr NAV, or (2) current NAV entry may be deleted (tx/rx could not send/receive CTS).
                        currTime = round(self.env.now) # in ns
                        newNAV = parameters.getMaxNAV(txID,currTime)-currTime   # If < 0, no NAV.
                        if newNAV > 0:
                            # Recheck the NAV after each slot.
                            yield self.env.timeout(max(0,min(newNAV,parameters.SLOT_DURATION)))
                            addDIFSDuration = True  # After NAV expires, this node should sense the channel again for DIFS duration.
                        else:   # NAV expired
                            if addDIFSDuration: # Sense channel for DIFS duration.
                                timeout += parameters.DIFS_DURATION
                            break
            
                    yield self.env.timeout(parameters.RADIO_SWITCHING_TIME*parameters.STEP_SIZE)  # Sh: 1 (default), increased duration to reduce computation time.
                    channel_sensing_froze_backoff = (self.env.now <= self.freeze_backoff_until)
                    if not channel_sensing_froze_backoff:
                        timeout -= parameters.RADIO_SWITCHING_TIME*parameters.STEP_SIZE   # Sh: 1 (default)

                    # sensing phase
                    if self.phy.isSending:   # I cannot sense while sending
                        if self.phy.transmission is not None: yield self.phy.transmission   # wait for my phy to finish sending other packets
                        timeout = parameters.DIFS_DURATION + backoff    # if a trasmission occours during the sensing I restart the sensing phase from scratch
                
                # AT: Calculating powers now that backoff is done, to use the latest interference from the receiver
                self.node.computePowers(macPkt.destination, macPkt.MCS_index)
                
                # Sh: With RTS/CTS functionality, Differentiate between CTS timeout and ACK timeout, and use NAV
                # CTS Timeout cases: Tx-Rx ndoes are not in range; Rx has an active NAV set which prevents it from responding with CTS.
                currTime = round(self.env.now) # in ns
                # print(f"mac.py: nav: no edge? ({not parameters.Current_Topology.has_edge(txID,rxID)}) or rxMaxNav {parameters.getMaxNAV(rxID,currTime)} > {currTime} -> {parameters.getMaxNAV(rxID,currTime)-currTime} or rxID not in listeningNodes {rxID not in [getNodeIdx(node.name) for node in self.ether.listeningNodes]}")
                cts_timeout = False
                
                has_edge: bool = parameters.Current_Topology.has_edge(txID,rxID) # type: ignore[reportUnknownMemberType]
                nav_is_set_at_rx: bool = parameters.getMaxNAV(rxID,currTime) > currTime
                rx_is_not_listening: bool = rxID not in [getNodeIdx(node.name) for _, node in self.ether.channelsAndListeningNodes]
                
                if ((not has_edge) or nav_is_set_at_rx or rx_is_not_listening):
                    cts_timeout = True
                    #ANDRES: now uses the mac's routing power and the destination's external interference
                    # NAV_duration = parameters.computeCtsTimeout(macPkt.MCS_index, self.routing_power, parameters.NODE_REGISTRY[macPkt.destination].external_interference)
                    # SD: ??? changed to self.transmission_power
                    # print("macPkt.MCS_index, self.transmission_power = ", macPkt.MCS_index, self.transmission_power)
                    
                    NAV_duration = parameters.computeCtsTimeout(self.node.tx_power_rts)
                    # NAV_duration = parameters.computeCtsTimeout(rts_power, parameters.NODE_REGISTRY[macPkt.destination].external_interference)
                    # NAV_duration += parameters.SLOT_DURATION if NAV_duration % parameters.SLOT_DURATION != 0 else 0
                    macPkt.NAV = NAV_duration
                    if parameters.PRINT_LOGS:
                        print("Case 1 at node:", self.name, "time:", currTime, "pkt id:", macPkt.id, "CTS Timeout NAV duration:", NAV_duration, "Curr time:", currTime, f"{has_edge=}, {nav_is_set_at_rx=}, {rx_is_not_listening=}")
                    
                    # Sh: Update the NAV entry at each 1-hop neighbor node (based on parameters.CONTROL_MCS_INDEX) in range of tx node
                    # print("Case 1 at node:",self.name, "time:", currTime, "pkt id:", macPkt.id)
                    #parameters.addNAVEntryAtAllOneHopNghbrs(txID,rxID,macPkt.id,round(currTime+NAV_duration),currTime,False)
                    self.addNAVEntryAtAllOneHopNghbrs(txID, rxID, macPkt, round(currTime+NAV_duration), currTime, RTS_flag=True)  # SD: ??? Add RTS_flag
                else:   # Use ACK timeout
                    
                    
                    NAV_duration = parameters.computeAckTimeout(self.node.tx_power_rts, macPkt.tx_power, int((macPkt.length-parameters.MAC_HEADER_LENGTH)//parameters.BASE_PAYLOAD_LENGTH))
                    # NAV_duration = parameters.computeAckTimeout(rts_power, cts_power, parameters.NODE_REGISTRY[macPkt.destination].external_interference, int((macPkt.length-parameters.MAC_HEADER_LENGTH)//parameters.BASE_PAYLOAD_LENGTH))
                    # print(f"{NAV_duration} += {parameters.SLOT_DURATION} if {NAV_duration} % {parameters.SLOT_DURATION} = {NAV_duration % parameters.SLOT_DURATION} != 0")
                    # NAV_duration += parameters.SLOT_DURATION if (NAV_duration) % parameters.SLOT_DURATION != 0 else 0
                    
                    macPkt.NAV = NAV_duration + (parameters.SIFS_DURATION * macPkt.MCS_index) # INFO: Tested for MCS 0 and 1, computeAckTimeout already has 1 SIFS this isn't the same as in handleReceivedPacket() # AT: Added for transmission over small distance (650)
                    
                    if parameters.PRINT_LOGS:
                        print("Case 2 at node:", self.name, "time:", currTime, "pkt id:", macPkt.id, "ACK Timeout NAV duration:", NAV_duration, "Curr time:", currTime)

                    # Sh: Update the NAV entry at each 1-hop neighbor node (based on parameters.CONTROL_MCS_INDEX) in range of tx and rx nodes
                    # print("Case 2 at node:",self.name, "time:", currTime, "pkt id:", macPkt.id)
                    #parameters.addNAVEntryAtAllOneHopNghbrs(txID,rxID,macPkt.id,round(currTime+NAV_duration),currTime,False)   # 1-hop neighbors of transmitter
                    self.addNAVEntryAtAllOneHopNghbrs(txID, rxID, macPkt, round(currTime+NAV_duration), currTime, RTS_flag=True)  # SD: ??? Add RTS_flag  # 1-hop neighbors of transmitter
                    
                    # print("Case 3 at node:",self.name, "time:", currTime, "pkt id:", macPkt.id)
                    #parameters.addNAVEntryAtAllOneHopNghbrs(rxID,txID,macPkt.id,round(currTime+NAV_duration),currTime)   # 1-hop neighbors of receiver
                    self.addNAVEntryAtAllOneHopNghbrs(rxID, txID, macPkt, round(currTime+NAV_duration), currTime)  # 1-hop neighbors of receiver (CTS sent with rx node's defaultrouting power and parameters.CONTROL_MCS_INDEX)
                if parameters.PRINT_LOGS: 
                    print("After adding new NAV entry: node: ", self.name, "NAV Table:", parameters.NAV_Table,"\n")

                self.phy.send(macPkt)
                self.pendingPackets[macPkt.id] = self.env.process(self.waitAck(macPkt, cts_timeout))
                self.isSensing = False
                # print(f"[MAC], waitIdleAndSend: out, @{self.env.now}, {self.name}, {self.hasPacket = }, {self.pendingPackets = }, {len(self.retransmissionCounter)} | {self.retransmissionCounter = }")
                return
            except simpy.Interrupt:
                phy_froze_backoff = (self.env.now <= self.freeze_backoff_until)
                
                if backoff == 0:    # need to add backoff, even if this is not a retransmission
                    backoff = int(random.random() * parameters.CW_MIN) * parameters.SLOT_DURATION
                elif timeout > backoff: # backoff has not been consumed, new timeout is DIFS + backoff
                    pass
                else:   # backoff has been partially consumed, new timeout is DIFS + remaining backoff
                    backoff = timeout
                
                timeout = backoff
                
                if not phy_froze_backoff:
                    timeout += parameters.DIFS_DURATION
            continue
