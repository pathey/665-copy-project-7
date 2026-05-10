from dataclasses import dataclass

@dataclass(slots=True)
class MacPacket:
    source: str
    destination: str
    length: int    # in bit
    id: str
    ack: bool # Sh: Added for simulation
    tx_power: float    # Sh: in watts (Used at transmitter only, not at receiver. Not included in MAC layer packets)
    intf_power: float # AT: Interference in watts (Used only in RTS CTS)
    rtg_power: float # Andres: Not used TODO
    # ext_intf: float # Andres: Not used 
    MCS_index: int  # AT: Transmitter notifies the MCS to use for its (combined) data packet. Same MCS for each data packet in a combined packet. Control packets use parameters.CONTROL_MCS_INDEX
    route_index: int # Andres: Not used currently (Used in old congestion detection algorithm)
    
    aggregated_ack_status: list[bool] # AT: Added only in ACK packets to notify reception status of each packet in a combined data packet
    
    NAV: float = 0    # Sh: NAV duration (Used at transmitter only, not at receiver. Not included in MAC layer packets)

    # FinishedSending: Optional[simpy.Event] = None