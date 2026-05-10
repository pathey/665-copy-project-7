'''
# Sh: File contains transport layer funcitonality to reinsert the packets dropped due to being stuck at the intermediate nodes when the route changes
# last modified on: Aug 6, 2024
'''
from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVar
import numpy as np
import parameters
import RouteFunctions as rf


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

# Remove packets belonging to this flow ID from each of the obsolete nodes and reinsert them at the flow source.
@profile
def ReinsertPacketsAtFlowSrc(flowID: str, obsoleteNodes: list[int], allNodes: list["Node"]) -> None:
    # Get flow source and destination nodes
    flowSrc = (flowID.split("_"))[0]; flowDest = (flowID.split("_"))[1]
    
    for eachNode in allNodes:
        if rf.getNodeIdx(eachNode.name) in obsoleteNodes:
            queue = eachNode.q; qSize = queue.qsize()
            
            # Iterate through each packet to check its flow id
            for _ in range(qSize):
                curr_pkt = queue.get()

                if (flowID in curr_pkt[3]): # Found relevant packet
                    # Record packet drop at this node
                    if parameters.PRINT_LOGS:
                        print('In TransportLayer.py: Time %d: Cannot forward packet %s. Outdated node %s. Drop from its queue! Resinsert at source %s. \n' % (eachNode.env.now, curr_pkt[3], eachNode.name, flowSrc))
                    intf_from_registry = parameters.INTF_Registry[rf.getNodeIdx(eachNode.name)]
                    _ = eachNode.stats.loqQueueStats(eachNode.name, curr_pkt[3], 0, float(curr_pkt[3].split('_')[0]), eachNode.env.now, eachNode.mac.freeze_counter, 0 if intf_from_registry is None else intf_from_registry, np.nan, parameters.PKT_DROPPED, parameters.DROP_REASON_NO_ROUTE)

                    # Reinsert the packet at the source node
                    ReinsertThisPacketAtFlowSrc(flowSrc, flowDest, curr_pkt[3], curr_pkt[2], allNodes)

                else: # Put the packet back
                    queue.put(curr_pkt)

            if parameters.PRINT_LOGS:
                print("In TransportLayer.py: Node name:", eachNode.name, "before queue size:", qSize, "after queue size:", eachNode.q.qsize()) 
    return


# Reinsert the pkt at flow source. If allNodes is None, flowSrc is the pointer to the flow source, otherwise it is the node name. Use the node name and allNodes to access the node's queue.
# TODO: This function can also be used to reinsert the packets dropped due to other reasons retry limit, overflow (not at source node), and queue management.
@profile
def ReinsertThisPacketAtFlowSrc(flowSrc: str, flowDest: str, flowBasedID: str, pktLength: int, allNodes: Optional[list["Node"]] = None) -> None:
    # Increment the packet reinsertion count
    currReinsertionCount, rem_pktID = rf.getTimesPktReinserted(flowBasedID)
    try:
        currReinsertionNumber = parameters.NUMBERS_TO_WORDS.index(currReinsertionCount)
        newFlowBasedID = rem_pktID + "_" + parameters.NUMBERS_TO_WORDS[currReinsertionNumber+1]
    except ValueError:
        print(f"In TransportLayer.py: Cannot reinsert packet {flowBasedID} as it has exceeded the total packet reinsertion limit ({parameters.NUMBERS_TO_WORDS[-1]})!")
        return
    
    if allNodes is not None:
        for eachNode in allNodes:
            if (parameters.NODE_REGISTRY[flowSrc].name == eachNode.name):
                # qSize_before = eachNode.q.qsize()
                eachNode.EnqueuePacket(pktLength, newFlowBasedID, flowDest) # Re-insert the packet
                #if parameters.PRINT_LOGS:
                #    print("In TranportLayer.py: Case 1: node:", eachNode.name, "pkt id:",newFlowBasedID, "length:", pktLength, "flow dest:", flowDest, "Before queue size:", qSize_before, "After queue size:", eachNode.q.qsize())
                break
    else:
        # qSize_before = flowSrc.q.qsize()
        parameters.NODE_REGISTRY[flowSrc].EnqueuePacket(pktLength, newFlowBasedID, flowDest)  # Re-insert the packet
        #if parameters.PRINT_LOGS:
        #    print("In TranportLayer.py: Case 2: node:", flowSrc.name, "pkt id:",newFlowBasedID, "length:", pktLength, "flow dest:", flowDest, "Before queue size:", qSize_before, "After queue size:", flowSrc.q.qsize())
    return