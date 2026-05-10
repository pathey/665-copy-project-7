from collections import defaultdict

import numpy as np
from macPacket import MacPacket
import simpy

from phyPacket import PhyPacket
import parameters

# Sh
import RouteFunctions as rf
from RouteFunctions import getNodeIdx
from ether import Ether, computeDistance

from typing import TYPE_CHECKING, Callable, ParamSpec, TypeVar

from simpy import Process
from simpy.events import ProcessGenerator

if TYPE_CHECKING:
    from mac import Mac

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

interference_files = {}

@profile
def cleanup_slot_information(env: simpy.Environment):
    all_nodes_phy = [node.mac.phy for node in list(parameters.NODE_REGISTRY.values())]
    timeout_duration = parameters.SLOT_DURATION
    while True:
        start_of_timeslot = env.now
        yield env.timeout(timeout_duration) # Wait for slot to end before reset
        for each_node in all_nodes_phy:
            each_node.start_of_timeslot = start_of_timeslot

            if each_node.channel_busy:
                each_node.mac.freeze_counter += 1
                if parameters.PRINT_LOGS:
                    print(f"phy.py: Froze backoff for {each_node.name} at slot: {start_of_timeslot} until {each_node.mac.freeze_backoff_until}")
            
            each_node.total_power_received.add(each_node.received_power_current_timeslot_data)
            for d in each_node.received_power_from:
                if not d[4].macPkt.ack:
                    each_node.power_from_others[d[1]].add(d[3])
                    if d[4].macPkt.destination != each_node.name:
                        each_node.total_intf_received.add(d[3])
                        each_node.intf_from_others[d[1]].add(d[3])

            # global interference_files
            # if self.mac.node.name not in interference_files and self.mac.node.name not in ["Node0", "Node11"]:
            #     interference_files[self.mac.node.name] = open(f"{parameters.RESULTS_FOLDER}/interference_{self.mac.node.name}.txt", "a", buffering=64*1000)
            #     atexit.register(interference_files[self.mac.node.name].close)
            # if self.mac.node.name not in ["Node0", "Node11"]:
            #     interference_files[self.mac.node.name].write(f"Now: {int(self.env.now)}, Slot: {int(self.start_of_timeslot)}, Total: {self.received_power_current_timeslot:.6e}, From: {self.received_power_from}\n")
            
            # DONE: 10/23 At cleanup, instead of the power, save the interference observed because different packets in the queue may be going to different places
            for _, _, _, eachReceivedPower, eachReceivedPkt, detectable in each_node.received_power_from:
                if detectable:
                    each_node.calculateInstantaneousSINR(eachReceivedPkt, eachReceivedPower)
            
            if each_node.received_power_current_timeslot == 0:
                parameters.INTF_Registry[each_node.id] = each_node.received_power_last_timeslot
            else:
                parameters.INTF_Registry[each_node.id] = each_node.received_power_current_timeslot
            each_node.received_power_last_timeslot = each_node.received_power_current_timeslot
            each_node.received_power_current_timeslot = 0
            each_node.received_power_current_timeslot_data = 0
            each_node.received_power_from.clear()
            if (env.now <= each_node.mac.freeze_backoff_until):
                each_node.channel_busy = False

@profile
def update_intf_table(env: simpy.Environment):
    all_nodes_phy = [node.mac.phy for node in list(parameters.NODE_REGISTRY.values())]
    timeout_duration = parameters.SIFS_DURATION
    
    while True:
        for each_node in all_nodes_phy:
            if each_node.received_power_current_timeslot == 0:
                parameters.INTF_Registry[each_node.id] = each_node.received_power_last_timeslot
            else:
                parameters.INTF_Registry[each_node.id] = each_node.received_power_current_timeslot

        yield env.timeout(timeout_duration)

# Summing algorithm from: https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Precision
class OnlineStats:
    __slots__ = ("sum", "c", "count", "min", "max")

    def __init__(self) -> None:
        self.sum = 0.0
        self.c = 0.0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")

    def add(self, x: float) -> None:
        # Neumaier compensated sum
        t = self.sum + x
        if abs(self.sum) >= abs(x):
            self.c += (self.sum - t) + x
        else:
            self.c += (x - t) + self.sum
        self.sum = t

        # other stats
        self.count += 1
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def mean(self):
        return (self.sum + self.c) / self.count if self.count else np.nan
class Phy(object):
    @profile
    def __init__(self, mac: 'Mac'):
        self.mac: 'Mac' = mac
        self.env: simpy.Environment = self.mac.env
        self.name: str = self.mac.name
        self.id: int = getNodeIdx(self.name)
        self.ether: Ether = self.mac.ether
        self.latitude: float = self.mac.latitude
        self.longitude: float = self.mac.longitude

        # AT: 2/16/26: Access powers at node level. Having a primitive that must be updated at all layers is prone to error.
        # self.tx_power_data = self.mac.tx_power_data
        # self.tx_power_ack = self.mac.tx_power_ack
        # self.tx_power_rts = self.mac.tx_power_rts
        # self.tx_power_cts = self.mac.tx_power_cts
        
        self.external_interference = self.mac.external_interference
        self.routing_power = self.mac.routing_power

        self.listen_process: Process = self.env.process(self.listen())
        self.receivingPackets: list[tuple[PhyPacket, float]] = []
        self.isSending = False  # keep radio state (Tx/Rx)
        self.transmission = None    # keep the transmitting process
        self.stats = self.mac.stats # Sh: To access/update stats of a node
        
        self.start_of_timeslot: float = self.env.now
        self.received_power_last_timeslot: float = 0
        self.received_power_current_timeslot: float = 0
        self.received_power_current_timeslot_data: float = 0
        self.received_power_from: list[tuple[int, str, str, float, PhyPacket, bool]] = []
        self.channel_busy = False
        
        self.total_power_received: OnlineStats = OnlineStats()
        self.power_from_others: dict[str, OnlineStats] = defaultdict(OnlineStats)
        self.total_intf_received: OnlineStats = OnlineStats()
        self.intf_from_others: dict[str, OnlineStats] = defaultdict(OnlineStats)
        self.tracker_power_rts: OnlineStats = OnlineStats()
        self.tracker_power_cts: OnlineStats = OnlineStats()
        self.tracker_power_data: OnlineStats = OnlineStats()
        self.tracker_power_ack: OnlineStats = OnlineStats()
        self.receiver_drops: int = 0
        self.receiver_dropped_packet: int = 0

    @profile
    def send(self, macPkt: MacPacket) -> None:
        #print("phy.py: send at node:",self.name)
        
        if not self.isSending:  # I do not send if I'm already sending
            self.listen_process.interrupt(macPkt)

    @profile
    def encapsulateAndTransmit(self, macPkt: MacPacket):
        # print("phy.cc: encapsulate and transmit at node:",self.name)
        self.receivingPackets.clear() # I switch to transmitting mode, so I drop all ongoing receptions
        yield self.env.timeout(parameters.RADIO_SWITCHING_TIME) # simulate time of radio switching
        self.ether.removeInChannel(self.inChannel, self.mac.node)
        # print("Time " + str(self.env.now) + " Node " + self.name + " " + macPkt.id + " exits channel")
        # yield self.env.timeout(parameters.RADIO_SWITCHING_TIME) # simulate time of radio switching # Sh: Commented since we have already switched two lines above. Bug?

        if parameters.PRINT_LOGS:
            print('phy.py: Time %d: %s stops listening' % (self.env.now, self.name))

        # Sh: Transmission delay increases when RTS/CTS packets are transmitted. Note: We do not actually transmit RTS/CTS.
        # When receiver node is out of range of the transmitter node or its NAV is still set, transmit only RTS packet. At MAC, wait for CTS timeout. 
        currTime = round(self.env.now)
        macPktConnected: bool = parameters.Current_Topology.has_edge(getNodeIdx(macPkt.source), getNodeIdx(macPkt.destination))
        if ((not macPktConnected) or (parameters.getMaxNAV(getNodeIdx(macPkt.destination), currTime) > currTime) or (rf.getNodeIdx(macPkt.destination) not in [rf.getNodeIdx(node.name) for _, node in self.ether.channelsAndListeningNodes])):
            pktLength = parameters.RTS_LENGTH + parameters.PHY_HEADER_LENGTH
            # duration = pktLength * parameters.getBitTxRate(macPkt.MCS_index)
            # SD: ??? Using default parameters.CONTROL_MCS_INDEX for RTS and CTS
            duration = pktLength * parameters.getBitTxRate(parameters.CONTROL_MCS_INDEX)
            if parameters.PRINT_LOGS:
                if (not macPktConnected):
                    print(f"phy.py: {self.env.now} Tx-Rx node pair are not in range. Use CTS timeout for packet: {macPkt.id}!")
                elif (parameters.getMaxNAV(getNodeIdx(macPkt.destination), currTime) > currTime):
                    print(f"phy.py: {self.env.now} Rx has its NAV set. Use CTS timeout for packet: {macPkt.id}!")
                elif ((rf.getNodeIdx(macPkt.destination) not in [rf.getNodeIdx(node.name) for _, node in self.ether.channelsAndListeningNodes])):
                    print(f"phy.py: {self.env.now} Rx is not listening. Use CTS timeout for packet: {macPkt.id}!")
                else:
                    print(f"phy.py: {self.env.now} Should not reach here")
        
        else:   # Sh: Otherwise send data or ack packet. MAC packet length already includes the MAC header leangth. Do not inlcude MAC header in RTS, CTS and ACK.            
            if (macPkt.ack):    # ACK packet
                pktLength = macPkt.length + parameters.PHY_HEADER_LENGTH
                duration = pktLength * parameters.getBitTxRate(parameters.CONTROL_MCS_INDEX) # AT: ACK Packets are sent at parameter.CONTROL_MCS_INDEX; Also check handleReceivedPacket() in mac.py.
                if parameters.PRINT_LOGS:
                    print("MAC:", macPkt.length, "PKT:", pktLength)
            
            else:   # Data packet. Add delay incurred while sending RTS/CTS and SIFS wait
                rts_cts_pktLength = parameters.RTS_LENGTH + parameters.CTS_LENGTH + 2 * parameters.PHY_HEADER_LENGTH
                pktLength = macPkt.length + parameters.PHY_HEADER_LENGTH #/parameters.MCS_SNR_TABLE[macPkt.MCS_index]['Coding']
                duration = rts_cts_pktLength * parameters.getBitTxRate(parameters.CONTROL_MCS_INDEX) + \
                    pktLength * parameters.getBitTxRate(macPkt.MCS_index) + \
                    2 * parameters.SIFS_DURATION
                if parameters.PRINT_LOGS:
                    print("RTS:", parameters.RTS_LENGTH, "CTS:", parameters.CTS_LENGTH, "2xPHY:", 2*parameters.PHY_HEADER_LENGTH,\
                      "RTSCTS:", rts_cts_pktLength, "CtrlBitTx:", parameters.getBitTxRate(parameters.CONTROL_MCS_INDEX), "PKT:", pktLength, "MAC:", macPkt.length,\
                      "BitTxRate:", parameters.getBitTxRate(macPkt.MCS_index), "2SIFS:", 2*parameters.SIFS_DURATION,\
                    )
                
        
        if parameters.PRINT_LOGS:
            print(f"phy.py: Time: {currTime}, at node: {self.name}, packet id: {macPkt.id+("ACK" if macPkt.ack else "")}, duration: {duration}")
        # Sh: End
        
        # Log Time and Power pkt was sent at.
        
        # interference transmitter observed, distance, mcs, transmitter's transmit power

        phyPkt = PhyPacket( False, macPkt) # start of packet
        if macPkt.ack:
            if parameters.PRINT_LOGS:
                print('phy.py: Time %d: %s PHY starts transmission of %s ACK with power %f' % (self.env.now, self.name, phyPkt.macPkt.id, macPkt.tx_power))
        else:
            if parameters.PRINT_LOGS:
                print('phy.py: Time %d: %s PHY starts transmission of %s with power %f' % (self.env.now, self.name, phyPkt.macPkt.id, macPkt.tx_power))
        
        # AT: Functionality changed to account for physical sensing and to determine channel busy
        
        retransmitted_num = self.mac.retransmissionCounter.get(macPkt.id, 0)
        transmissions = 0
        if duration <= parameters.SLOT_DURATION:
            # Sh: Added retrun args, and one input arg (None)
            self.ether.transmit(phyPkt, self.latitude, self.longitude, True, True, self.received_power_current_timeslot, retransmitted_num) # beginOfPacket=True, endOfPacket=True
            transmissions += 1
            yield self.env.timeout(duration)  # AT: only send 1 signal per slot even if duration < slot  # wait only remaining time
            duration = 0
        else:
            # Sh: Added retrun args, and one input arg (None)
            self.ether.transmit(phyPkt, self.latitude, self.longitude, True, False, self.received_power_current_timeslot, retransmitted_num) # beginOfPacket=True, endOfPacket=False
            transmissions += 1
            yield self.env.timeout(parameters.SLOT_DURATION) # send a signal every slot
            duration -= parameters.SLOT_DURATION
            while True:
                if duration <= parameters.SLOT_DURATION:
                    # Sh: Added input arg (destination channels considered at the begin of packet)
                    self.ether.transmit(phyPkt, self.latitude, self.longitude, False, True, self.received_power_current_timeslot, retransmitted_num)  # beginOfPacket=False, endOfPacket=True
                    transmissions += 1
                    yield self.env.timeout(duration)  # AT: only send 1 signal per slot even if duration < slot  # wait only remaining time
                    duration = 0
                    break
                else:                    
                    # Sh: Added input arg (destination channels considered at the begin of packet)
                    self.ether.transmit(phyPkt, self.latitude, self.longitude, False, False, self.received_power_current_timeslot, retransmitted_num)  # beginOfPacket=False, endOfPacket=False
                    transmissions += 1
                    yield self.env.timeout(parameters.SLOT_DURATION) # send a signal every slot
                    duration -= parameters.SLOT_DURATION
                if parameters.PRINT_LOGS:
                    print("Remaining duration:", duration, "curr time:",self.env.now, "pkt id:",macPkt.id, "node name:", self.name)
        
        # if macPkt.finished_transmission is not None and not macPkt.finished_transmission.triggered:
        #     macPkt.finished_transmission.succeed()

        if macPkt.ack:
            if parameters.PRINT_LOGS:
                print('phy.py: Time %d: %s PHY ends transmission of %s ACK with %s transmissions' % (self.env.now, self.name, phyPkt.macPkt.id, transmissions))
        else:
            if parameters.PRINT_LOGS:
                print('phy.py: Time %d: %s PHY ends transmission of %s with %s transmissions' % (self.env.now, self.name, phyPkt.macPkt.id, transmissions))

        self.inChannel = self.ether.getInChannel(self.mac.node)
        # print("Time " + str(self.env.now) + " Node " + self.name + " " + macPkt.id + " enters back into channel")
        yield self.env.timeout(parameters.RADIO_SWITCHING_TIME) # simulate time of radio switching
        if parameters.PRINT_LOGS:
            print('phy.py: Time %d: %s starts listening' % (self.env.now, self.name))
            

            
    # @profile
    def listen(self) -> ProcessGenerator:
        # print("phy.cc: listen at node:",self.name)
        self.inChannel = self.ether.getInChannel(self.mac.node)
        # print("Time " + str(self.env.now) + " Node " + self.name + " enters channel")
        yield self.env.timeout(parameters.RADIO_SWITCHING_TIME) # simulate time of radio switching
        if parameters.PRINT_LOGS:
            print('phy.py: Time %d: %s starts listening' % (self.env.now, self.name))

        while True:
            try:
                phyPkt: PhyPacket
                beginOfPacket: bool
                endOfPacket: bool
                
                (phyPkt, beginOfPacket, endOfPacket) = yield self.inChannel.get()
                
                if parameters.PRINT_LOGS:
                    print(f'phy.py: Time {self.env.now}: {self.name} {"detects" if phyPkt.power[self.name] + parameters.NOISE_FLOOR > parameters.RADIO_SENSITIVITY else "does not detect"}' +
                          f' {("BEGIN" if beginOfPacket else "") + ("END" if endOfPacket else "") + phyPkt.macPkt.id + ("ACK" if phyPkt.macPkt.ack else "")}' + 
                          f' from {phyPkt.macPkt.source} with power {phyPkt.power[self.name]:.6e} transmitted at {phyPkt.macPkt.tx_power:.6f} at MCS:{phyPkt.macPkt.MCS_index}')
                
                # If we have a packet to send, trigger decide whether to freeze the counter. Else, reciever performs steps 1, 2. When it sends packet to mac, subtract phyPackt.power[self.name] from total interference. Then mac will do the remaining processing.
                # 1. Compute total interference
                # 2. Compute minimum recieved power for parameters.CONTROL_MCS_INDEX and observed total interference in step 1, noise is known
                # 3. Identify distance from the packet we are transmitting using the reciever node and its location (node.py)
                # 4. Compute minimum transmit power required to achieve minimum recieved power (step 2) # Store this value; use this to transmit packet if table is empty, else use the value in table, and add to macPacket
                # 5. If this power is lower than maximum transmission power, then consider channel to be free. Else, channel is busy, freeze counter if needed.

                # Sum recieved powers, reset after slot duration
                # # For channel busy, use whatever power we have now, even if we get more power later instead of previous timeslot
                # For RTS and CTS, use the power in the last timeslot
                    
                # Why are there cases where there is a packet from a node multiple times here?
                # Check how inChannel.get works, could the function be getting interrupted and double counting when it runs again?
                self.received_power_current_timeslot += phyPkt.power[self.name]
                if not phyPkt.macPkt.ack:
                    self.received_power_current_timeslot_data += phyPkt.power[self.name]
                
                # Note: Things like phyPkt.power[self.name] might change outside of this function since the same phyPkt object is reused
                self.received_power_from.append((int(self.env.now), phyPkt.macPkt.source, phyPkt.macPkt.id, phyPkt.power[self.name], phyPkt, phyPkt.power[self.name] + parameters.NOISE_FLOOR > parameters.RADIO_SENSITIVITY))    

                if self.mac.hasPacket and len(self.mac.retransmissionCounter) > 0:
                    macpkt_id = next(iter(self.mac.retransmissionCounter))
                    sending_to_node_id = macpkt_id.split('_')[2]
                    sending_to_node = parameters.NODE_REGISTRY[sending_to_node_id]
                    distance_to_reciever = computeDistance(self.mac.node.latitude, self.mac.node.longitude, sending_to_node.latitude, sending_to_node.longitude)
                    
                    # Determine if the channel is busy using recieved power in the current timeslot
                    min_transmit_power = parameters.getTransmitPower(distance=distance_to_reciever, mcs_index=phyPkt.macPkt.MCS_index, noise=parameters.NOISE_FLOOR, 
                                                                    interference=self.received_power_current_timeslot)
                    max_transmit_power = parameters.getMaximumTransmitPower(phyPkt.macPkt.MCS_index)
                    self.channel_busy = min_transmit_power >= max_transmit_power
                    
                    # DONE: For each node, track the number of times the backoff was frozen for each packet.
                    # DONE: 10/23 This can be moved up to the cleanup
                    if self.channel_busy:
                        self.mac.freeze_backoff_until = self.start_of_timeslot + parameters.SLOT_DURATION + parameters.DIFS_DURATION

                    # if parameters.PRINT_LOGS:
                    #     print(f"phy.py: {self.mac.node.name}: t={int(self.env.now)} pkt:{phyPkt.macPkt.id+("ACK" if phyPkt.macPkt.ack else "")} Calc Transmit Power: {min_transmit_power} >= {max_transmit_power}, Channel Busy: {self.channel_busy}, interference: {self.received_power_current_timeslot}, freeze until {self.mac.freeze_backoff_until}")
                
                if phyPkt.power[self.name] + parameters.NOISE_FLOOR > parameters.RADIO_SENSITIVITY: # detected signal # if phyPkt.power[self.name] + parameters.NOISE_FLOOR > parameters.RADIO_SENSITIVITY

                    # the signal just received will interfere with other signals I'm receiving (and vice versa)
                    for receivingPkt, time in self.receivingPackets:
                        if receivingPkt != phyPkt:
                            #receivingPkt.interferingSignals[phyPkt.macPkt.id] = phyPkt.power[self.name]
                            #phyPkt.interferingSignals[receivingPkt.macPkt.id] = receivingPkt.power[self.name]

                            # What about when data packet and ack packet have different powers? 
                            # Sh: Packet is shared with all nodes. Therefore, update interfering signals for only this node.
                            if (self.name not in receivingPkt.interferingSignals): receivingPkt.interferingSignals[self.name] = {}
                            # if (phyPkt.macPkt.id not in receivingPkt.interferingSignals[self.name]): receivingPkt.interferingSignals[self.name][phyPkt.macPkt.id] = (phyPkt.power[self.name], self.env.now)
                            receivingPkt.interferingSignals[self.name][phyPkt.macPkt.id] = (phyPkt.power[self.name], self.env.now)
                            if (self.name not in phyPkt.interferingSignals): phyPkt.interferingSignals[self.name] = {}
                            # if (receivingPkt.macPkt.id not in phyPkt.interferingSignals[self.name]): phyPkt.interferingSignals[self.name][receivingPkt.macPkt.id] = (receivingPkt.power[self.name], time)
                            phyPkt.interferingSignals[self.name][receivingPkt.macPkt.id] = (receivingPkt.power[self.name], time)
                            
                            if parameters.PRINT_LOGS:
                                print(f"Node:{self.name}, Time: {self.env.now} Packet {phyPkt.macPkt.id + ("ACK" if phyPkt.macPkt.ack else "")} (power={phyPkt.power[self.name]}) is causing interference to {receivingPkt.macPkt.id + ("ACK" if receivingPkt.macPkt.ack else "")} (power={receivingPkt.power[self.name]})")
                    
                    # Sh: Brought it inside this if loop. Stop sensing only if received power is above radio sensitivity.
                    if self.mac.isSensing:  # interrupt mac if it is sensing for idle channel
                        #print("phy.py: sensing channel interrupted at %s in response to signal %s." % (self.name, phyPkt.macPkt.id))
                        if self.mac.sensing is not None: self.mac.sensing.interrupt()

                    if beginOfPacket:  # begin of packet
                        self.receivingPackets.append((phyPkt, self.env.now))
                    
                    # AT: Changed from elif to if for situations where packet duration is smaller than slot duration
                    if endOfPacket:   # end of packet
                        self.calculateInstantaneousSINR(phyPkt, phyPkt.power[self.name]) # At end of packet, we aren't waiting for end of slot to calculate sinr
                        if any(val[0] is phyPkt for val in self.receivingPackets): # consider only if I received the begin of the packet, otherwise I ignore it, as it is for sure corrupted
                            for i, val in enumerate(self.receivingPackets):
                                if val[0] is phyPkt:
                                    del self.receivingPackets[i]
                                    break

                            '''# Sh: For each remianing packets, remove phyPkt id from the interfering signals for this node. 
                            try:
                                for eachRemainPkt in self.receivingPackets:
                                    #for eachIntfSignal in eachRemainPkt.interferingSignals[self.name]:
                                    #    print(f"Before: Node:{self.name}, remain pkt:{eachRemainPkt}, intf signal:{eachIntfSignal}, intf signal power:{eachRemainPkt.interferingSignals[self.name][eachIntfSignal]}")
                                    if (phyPkt.macPkt.id in eachRemainPkt.interferingSignals[self.name]):
                                        del eachRemainPkt.interferingSignals[self.name][phyPkt.macPkt.id]
                                    #for eachIntfSignal in eachRemainPkt.interferingSignals[self.name]:
                                    #    print(f"After: Node:{self.name}, remain pkt:{eachRemainPkt}, intf signal:{eachIntfSignal}, intf signal power:{eachRemainPkt.interferingSignals[self.name][eachIntfSignal]}")                                    
                            finally:
                                if False: print("In phy.py: Node", self.name, "does not have any receiving packets yet!")'''
                            
                            if not phyPkt.corrupted:
                                sinr = self.computeSinr(phyPkt, 1 if phyPkt.macPkt.ack else len(phyPkt.macPkt.id.split('_COMBINED_'))) # AT: An ACK packet is a single packet even if it is for a combined packet
                                if parameters.PRINT_LOGS: 
                                    print("phy.py: node: %s, t=%s, rcvd packet (id = %s), sinr = %s, base sinr = %s" % (self.name, self.env.now, phyPkt.macPkt.id+("ACK" if phyPkt.macPkt.ack else ""), sinr, parameters.MCS_SNR_TABLE[phyPkt.macPkt.MCS_index]['MinSNR']))
                                
                                # Sh: Track interfering links in the neighborhood. Min signal SINR (at parameters.CONTROL_MCS_INDEX) is required to decode the RTS/CTS packet.
                                if max(sinr) >= parameters.MIN_MIN_REQUIRED_SINR:
                                    if not phyPkt.macPkt.ack:   # Tx->Rx link should be considered interfering link, not the Rx->Tx link. Tx sends Data packet. Rx sends Ack.
                                        linkID = f'{phyPkt.macPkt.source}-{phyPkt.macPkt.destination}'
                                        if linkID not in parameters.Node_Stats[getNodeIdx(self.name)]['IL']: 
                                            parameters.Node_Stats[getNodeIdx(self.name)]['IL'].add(linkID)
                                
                                # SD: ??? CTStimeout is calculated using phyPkt.macPkt.tx_power (Tx nodes transmission_power) instead of self.routing_power (which is Rx node's power)
                                # SD: original comment-(Sh: If Tx waits for CTS timeout, the received packet does not include payload, and Tx does not expect an ACK. Do not forward the received packet to the MAC layer.)
                                if (phyPkt.macPkt.NAV != parameters.computeCtsTimeout(parameters.NODE_REGISTRY[phyPkt.macPkt.source].tx_power_rts)): # AT: 2/14/26 Since RTS and Data are using different powers, we need to get the power of RTS since macpkt only has data's power
                                # if (phyPkt.macPkt.NAV != parameters.computeCtsTimeout(phyPkt.macPkt.tx_power)): # 2/12/26 Removed external interference
                                    # print("From PHY.py-------------------phyPkt.macPkt.MCS_index, phyPkt.macPkt.tx_power=",phyPkt.macPkt.ack, phyPkt.macPkt.MCS_index, phyPkt.macPkt.tx_power)
                                    # print("HERE--------",self.name, self.transmission_power, 
                                    #       phyPkt.macPkt.tx_power, round(phyPkt.macPkt.NAV,4) , 
                                    #        round(parameters.computeCtsTimeout(phyPkt.macPkt.MCS_index,phyPkt.macPkt.tx_power, parameters.NODE_REGISTRY[phyPkt.macPkt.destination].external_interference),4) )
                                    
                                    if max(sinr) >= parameters.MCS_SNR_TABLE[phyPkt.macPkt.MCS_index]['MinSNR']:    # signal greater than noise and inteference
                                        # The signal from this packet we recieved should not be considered as interference
                                        self.received_power_current_timeslot -= phyPkt.power[self.name]
                                        self.received_power_current_timeslot = min(0, self.received_power_current_timeslot)
                                        
                                        pkts_received = [val >= parameters.MCS_SNR_TABLE[phyPkt.macPkt.MCS_index]['MinSNR'] for val in sinr]
                                        self.env.process(self.mac.handleReceivedPacket(phyPkt.macPkt, sinr, pkts_received)) # Sh: Also pass sinr value
                                    
                                    # Sh: Logging insufficient packet SINR can help in debugging.
                                    elif ((phyPkt.macPkt.destination == self.name)):
                                        # TODO: If the collision is with a data packet (phyPkt.macPkt.ack), print packet, if I am one if the nodes in the first part of PacketID, increase collision count for the other node
                                        if parameters.PRINT_LOGS:
                                            # if not phyPkt.macPkt.ack:
                                            print(f"phy.py: At node: {self.name}, Insufficient SINR: {sinr} to decode the packet! With MCS: {phyPkt.macPkt.MCS_index}")
                                            print(self.received_power_from)
                                        
                                        # For debugging purpose, check the minimum sinr along the route. It does not matter if the receiving node is flow destination or not.
                                        pkt_ids = phyPkt.macPkt.id.split('_COMBINED_')
                                        if phyPkt.macPkt.ack:
                                            for i, pkt_id in enumerate(pkt_ids):
                                                self.stats.logDeliveredPacket(pkt_id, self.env.now, False, sinr[0])
                                            self.receiver_dropped_packet += 1
                                        else:
                                            for i, pkt_id in enumerate(pkt_ids):
                                                self.stats.logDeliveredPacket(pkt_id, self.env.now, False, sinr[i])
                                                self.receiver_dropped_packet += 1
                                        self.receiver_drops += 1

                            # Sh: Inform about discarding the corrupted packet
                            else:   
                                if ((parameters.PRINT_LOGS) and (phyPkt.macPkt.destination == self.name)):
                                    print('phy.py: Time %d: %s received a corrupted packet %s from %s. Discard!\n' % (self.env.now, self.name, phyPkt.macPkt.id, phyPkt.macPkt.source))
                
            except simpy.Interrupt as macPkt:        # listening can be interrupted by a message sending
                self.isSending = True
                if macPkt.cause is not None:
                    self.transmission = self.env.process(self.encapsulateAndTransmit(macPkt.cause))
                    yield self.transmission
                self.isSending = False

    @profile
    def calculateInstantaneousSINR(self, phyPkt: PhyPacket, phyPower: float):
        interference = 0
        if parameters.PRINT_LOGS:
            print(f"phy.py: INTFs At node: {self.name}: ", end='')
        for _, _, recID, recPower, recPkt, detectable in self.received_power_from:
            if not detectable: continue
            if recPkt is not phyPkt:
                interference += recPower
                if parameters.PRINT_LOGS:
                    print(f'{recID}: {recPower}, ', end='')
        if parameters.PRINT_LOGS:
            print('')
        sinr = phyPower/(interference + self.mac.external_interference + parameters.NOISE_FLOOR)
        
        if self.name not in phyPkt.instantaneousSINRs: phyPkt.instantaneousSINRs[self.name] = []
        phyPkt.instantaneousSINRs[self.name].append((self.env.now, sinr))
        if parameters.PRINT_LOGS:
            print(f"phy.py: At node: {self.name}, sinr: {sinr} for pkt: {phyPkt.macPkt.id + ("ACK" if phyPkt.macPkt.ack else "")} | {phyPower}/({interference}+{self.mac.external_interference}+{parameters.NOISE_FLOOR})")
    
    @profile
    def computeSinr(self, phyPkt: PhyPacket, nPkts: int):
        
        #print("phy.c: compute sinr at node:",self.name)
        interference = 0
        # Sh: Phy packet is shared with all nodes. Check interfering signals w.r.t. this node.
        
        # NOTE: When we have CRC, will this approach be useful?
        
        # This should be used for the LINK_QUALITY calculations used in route selection
        if self.name in phyPkt.interferingSignals:
            for interferingSignal in phyPkt.interferingSignals[self.name]:
                interference += float(phyPkt.interferingSignals[self.name][interferingSignal][0])
                # print(self.env.now, " At node:", self.name, "interference:", interference, "intf signal:", interferingSignal, "phypkt:", phyPkt.macPkt.id)
                if parameters.PRINT_LOGS: print("At node:", self.name, "intf sum:", interference, "intf signal:", interferingSignal, "intf:", phyPkt.interferingSignals[self.name][interferingSignal][0], "t:", phyPkt.interferingSignals[self.name][interferingSignal][1], "phypkt:", phyPkt.macPkt.id+("ACK" if phyPkt.macPkt.ack else ""))
                # TODO: In future, corrupt a particular portion of the packet which experienced interference. Let CRC at MAC fix the bit errors.
        
        nSINRs = len(phyPkt.instantaneousSINRs[self.name]) if self.name in phyPkt.instantaneousSINRs else 0
        minSinrs: list[float] = []
        if parameters.ENABLE_INDIVIDUAL_PKT_SINR:
            if phyPkt.instantaneousSINRs[self.name]:
                for i in range(nPkts):
                    start = (i * nSINRs) // nPkts
                    end = ((i + 1) * nSINRs) // nPkts
                    if start >= nSINRs:
                        start = nSINRs - 1
                    if end <= start:
                        minSinrs.append(phyPkt.instantaneousSINRs[self.name][start][1])
                    else:
                        bucket = phyPkt.instantaneousSINRs[self.name][start:end]
                        minSinrs.append(min(bucket, key=lambda x: x[1])[1])
        else:
            minSinrs = [min(phyPkt.instantaneousSINRs[self.name], key=lambda x: x[1])[1]] * nPkts
        
        # Sh: Update link quality stats for the received packet
        txNode = getNodeIdx(phyPkt.macPkt.source)
        rxNode = getNodeIdx(self.name)
        
        #print("\nIn phy.py: Before processing: ",rxNode, txNode, "at rx node", parameters.Link_Quality[rxNode],"\n", "at tx node: ", parameters.Link_Quality[txNode], "\n")
        # Update tx-rx link stats as observed at the rx node. Rx node = this node = destination
        if (phyPkt.macPkt.destination == self.name):
            try:
                if 'RcvdPackets' not in parameters.Link_Quality[rxNode][txNode]:
                    _ = str(txNode)+"-"+str(rxNode)
                    #print("Link %d-%d entry exists at %s!" % (txNode, rxNode, self.name))
            except:
                # Initialization
                parameters.Link_Quality[rxNode][txNode] = parameters.LinkQualityNeighborEntry({
                    'RcvdPackets': {'TotalPkts': 0,'AvgPowerRcvd':0.0,'AvgInterferingPower':0.0}, # Power in watts
                    'MCSIndex': parameters.CONTROL_MCS_INDEX,
                    'TxPower': phyPkt.macPkt.tx_power  # in watts
                })

            totalRcvdPkts = parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['TotalPkts']
            parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['TotalPkts'] += 1
            parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['AvgPowerRcvd'] = (phyPkt.power[self.name] + \
                                                                        parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['AvgPowerRcvd']*totalRcvdPkts) / (1+totalRcvdPkts)
            parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['AvgInterferingPower'] = (interference + \
                                                                        parameters.Link_Quality[rxNode][txNode]['RcvdPackets']['AvgInterferingPower']*totalRcvdPkts) / (1+totalRcvdPkts)
            if not phyPkt.macPkt.ack:   # Data packet
                parameters.Link_Quality[rxNode][txNode]['MCSIndex'] = phyPkt.macPkt.MCS_index
                parameters.Link_Quality[rxNode][txNode]['TxPower'] = phyPkt.macPkt.tx_power
        
        # Update avg interfering power and packets for inactive links
        totalIntfPkts = parameters.Link_Quality[rxNode]['InterferingPkts']['TotalPkts']
        parameters.Link_Quality[rxNode]['InterferingPkts']['TotalPkts'] += 1
        # Do not consider received power as that is not interference
        parameters.Link_Quality[rxNode]['InterferingPkts']['AvgPower'] = (interference + 
                                                parameters.Link_Quality[rxNode]['InterferingPkts']['AvgPower'] * totalIntfPkts)/(1+totalIntfPkts)
        #print("In phy.py: After processing: ",rxNode, txNode, "at rx node", parameters.Link_Quality[rxNode],"\n", "at tx node: ", parameters.Link_Quality[txNode], "\n")
        # ANDRES: Added the external interfernce value to the calculation of the SINR
        if parameters.PRINT_LOGS:
            print(f"phy.py: At node: {self.name}, pkt: {phyPkt.macPkt.id + ("ACK" if phyPkt.macPkt.ack else "")} SINRs: {phyPkt.instantaneousSINRs[self.name]}")
        return minSinrs

