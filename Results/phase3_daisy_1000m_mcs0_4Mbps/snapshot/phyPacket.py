from dataclasses import dataclass, field
from macPacket import MacPacket

@dataclass(slots=True)
class PhyPacket(object):
    corrupted: bool # Sh: Included in PHY_HEADER_LENGTH
    macPkt: MacPacket # Sh: Length of MAC frame (RTS, CTS, Data, ACK) is added separately
    power: dict[str, float] = field(default_factory=lambda: {}) # Sh, AT: Used only for SINR calculation locally
    interferingSignals: dict[str, dict[str, tuple[float, float]]] = field(default_factory=lambda: {}) # Sh, AT: Used only for SINR calculation locally
    instantaneousSINRs: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: {}) # Sh, AT: Used only for SINR calculation locally
    # numConcatPackets: bool