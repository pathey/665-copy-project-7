import os
import sys
import pandas as pd
import shutil
import re
from scipy import stats
import ast
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def move_directory(src, dst):
    print(src, dst)
    if not os.path.exists(dst):
        #shutil.copytree(src, dst)
        shutil.move(src, dst)
    else:
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                move_directory(s, d)
            else:
                #shutil.copy2(s, d)
                shutil.move(s, d)

def move_files(source_path, destination_path):
    # Ensure the destination path exists
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Iterate through each folder in the source path
    for folder_name in os.listdir(source_path):
        folder_path = os.path.join(source_path, folder_name); #print(folder_path)
        
        if os.path.isdir(folder_path):
            # Find trajectory folders (Trajectory<number>)
            for item in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, item)):      # e.g., results, Results
                    curr_folder = os.path.join(folder_path, item)
                    for eachItem in os.listdir(curr_folder):
                        if eachItem.startswith('Trajectory') and os.path.isdir(os.path.join(curr_folder, eachItem)):    
                            trajectory_folder_path = os.path.join(curr_folder, eachItem); #print(trajectory_folder_path)

                            # Check if the Trajectory folder is not empty
                            if os.listdir(trajectory_folder_path):
                                # Destination sub-folder for this trajectory
                                destination_trajectory_folder = os.path.join(destination_path, eachItem)

                                # Copy contents of the Trajectory folder to the destination
                                move_directory(trajectory_folder_path, destination_trajectory_folder)
                                print(f"Copied contents of {trajectory_folder_path} to {destination_trajectory_folder}")
    return

def get_pkt_drop_count(root, nodeId, flowSrc, flowDest):
    flowID = f"Node{flowSrc}_Node{flowDest}"; #print(flowID)
    fileName = os.path.join(root,f"Node{nodeId}.csv"); #print(fileName)
    dropped_count = {'Expiry': 0, 'QueueManagement':0, 'RetryLimit': 0, 'Overflow': 0, 'NoRoute':0}
    try:
        # Open the corresponding CSV file
        with open(fileName, mode='r') as file:
            reader = csv.DictReader(file)
                        
            # Iterate through each row in the CSV file
            for row in reader:
                #print(row)
                if int(row['Dropped?']) == 1:
                    drop_reason = row['Drop Reason']
                    if drop_reason in dropped_count:
                        if (row['Packet ID'].split('_')[1] == "Node"+str(flowSrc)) and (row['Packet ID'].split('_')[2] == "Node"+str(flowDest)):
                            dropped_count[drop_reason] += 1
            
    # Handle the case where the file does not exist
    except Exception as e:
        print(f"File {fileName} not found.")
    
    return [dropped_count['Expiry'], dropped_count['QueueManagement'], dropped_count['RetryLimit'], dropped_count['Overflow'], dropped_count['NoRoute']]

def calculate_instantaneous_pdr(df, time_window=1.0):
    """
    Calculate instantaneous PDR over time windows for a given flow.
    Uses Generation Time + Delay as the actual reception time.
    
    Args:
        df: DataFrame containing packet data with timestamps
        time_window: Time window in seconds for calculating instantaneous PDR (default: 1.0 second)
    
    Returns:
        DataFrame with columns: Time_Window, Generated_Packets, Received_Packets, Instantaneous_PDR
    """
    if df.empty:
        return pd.DataFrame(columns=['Time_Window', 'Generated_Packets', 'Received_Packets', 'Instantaneous_PDR'])
    
    # Get base packet IDs to handle TCP reinsertions
    def get_base_packet_id(packet_id):
        return re.match(r"(.+)_\w+$", packet_id).group(1)
    
    df_copy = df.copy()
    df_copy['Base Packet ID'] = df_copy['Pkt ID'].apply(get_base_packet_id)
    
    # Get the latest packet for each base packet ID
    df_latest = df_copy.groupby('Base Packet ID').tail(1).reset_index(drop=True)
    
    # Calculate actual reception time: Generation Time + Delay
    if 'Generation Time (in s)' in df_latest.columns and 'Delay (in s)' in df_latest.columns:
        df_latest['Reception Time (s)'] = df_latest['Generation Time (in s)'] + df_latest['Delay (in s)']
        # Filter out packets with infinite delay (not received)
        df_received = df_latest[(df_latest['Delay (in s)'] != 0) & (df_latest['Delay (in s)'] != float('inf'))]
        df_generated = df_latest  # All packets were generated
    else:
        # Fallback to original method if columns don't exist
        print("Warning: Generation Time or Delay columns not found, using fallback method")
        df_received = df_latest[(df_latest['Delay (in s)'] != 0) & (df_latest['Delay (in s)'] != float('inf'))]
        df_generated = df_latest
    
    if df_received.empty:
        return pd.DataFrame(columns=['Time_Window', 'Generated_Packets', 'Received_Packets', 'Instantaneous_PDR'])
    
    # Get time range for windows
    if 'Reception Time (s)' in df_received.columns:
        min_time = min(df_received['Reception Time (s)'])
        max_time = max(df_received['Reception Time (s)'])
    else:
        # Fallback: use generation time if available
        if 'Generation Time (in s)' in df_generated.columns:
            min_time = min(df_generated['Generation Time (in s)'])
            max_time = max(df_generated['Generation Time (in s)'])
        else:
            # Last resort: estimate time
            total_packets = len(df_generated)
            estimated_duration = 100  # seconds - adjust based on your simulation
            min_time = 0
            max_time = estimated_duration
    
    # Create time windows in seconds
    time_windows = np.arange(min_time, max_time + time_window, time_window)
    
    instantaneous_pdr_data = []
    
    for i in range(len(time_windows) - 1):
        start_time = time_windows[i]
        end_time = time_windows[i + 1]
        
        # Count packets generated in this time window
        if 'Generation Time (in s)' in df_generated.columns:
            generated_packets = len(df_generated[(df_generated['Generation Time (in s)'] >= start_time) & 
                                               (df_generated['Generation Time (in s)'] < end_time)])
        else:
            # Estimate based on packet index
            total_packets = len(df_generated)
            window_start_idx = int(start_time * total_packets / max_time)
            window_end_idx = int(end_time * total_packets / max_time)
            generated_packets = window_end_idx - window_start_idx
        
        # Count packets received in this time window
        if 'Reception Time (s)' in df_received.columns:
            received_packets = len(df_received[(df_received['Reception Time (s)'] >= start_time) & 
                                             (df_received['Reception Time (s)'] < end_time)])
        else:
            # Fallback: use generation time + delay if available
            if 'Generation Time (in s)' in df_received.columns and 'Delay (in s)' in df_received.columns:
                df_received['Reception Time (s)'] = df_received['Generation Time (in s)'] + df_received['Delay (in s)']
                received_packets = len(df_received[(df_received['Reception Time (s)'] >= start_time) & 
                                                 (df_received['Reception Time (s)'] < end_time)])
            else:
                received_packets = 0
        
        instantaneous_pdr = received_packets / max(1, generated_packets)
        
        instantaneous_pdr_data.append({
            'Time_Window': f"{start_time:.1f}-{end_time:.1f}",
            'Generated_Packets': generated_packets,
            'Received_Packets': received_packets,
            'Instantaneous_PDR': instantaneous_pdr
        })
    
    return pd.DataFrame(instantaneous_pdr_data)

def get_stats(parent_dir: Path, traj_number: str):
    folder_path: Path = parent_dir/traj_number
    exp_total_generated_pkts = 0
    exp_total_rcvd_pkts = 0
    exp_pdr = 0
    exp_avg_delay = 0
    exp_total_retransmissions = 0
    exp_total_forwards = 0
    exp_route_length = 0
    exp_route_uptime = 0
    exp_hq_route_uptime = 0
    exp_route_switches = 0
    total_exp = 0
    records = pd.DataFrame(columns=['File Path','Run No','Packets Generated','Packets Received','PDR','Avg Delay','#Retransmissions',\
                                    '#Forwards','Avg Route Length','%Route Uptime', '%HQ Route Uptime','Route Switches',\
                                    'Packet Drop (Expiry)','Packet Drop (Queue Management)','Packet Drop (Retry)','Packet Drop (Overflow)',\
                                    'Packet Drop (No Route)','Packet Reinserted (TCP)'])
    
    # Dictionary to store instantaneous PDR data for each flow
    instantaneous_pdr_data = defaultdict(list)
    
    # Iterate through each subfolder and file in the given folder
    for root, dirs, files in os.walk(folder_path):

        for file in files:
            if ((file.endswith('.csv')) and ('Flow_Node' in file)):
                file_path = os.path.join(root, file);# print(file_path, file)

                # Read the CSV file
                df = pd.read_csv(file_path)

                '''Start'''
                # If a packet is reinserted at source (TCP protocol), consider total packets generated and received for the last most entry
                df_TCP = df.copy()
                def get_base_packet_id(packet_id):
                    return re.match(r"(.+)_\w+$", packet_id).group(1)
                df_TCP['Base Packet ID'] = df_TCP['Pkt ID'].apply(get_base_packet_id); #print(df_TCP.shape)
                
                # Group by the base packet ID and get the last entry for each group
                df_TCP_latest_packets = df_TCP.groupby('Base Packet ID').tail(1).reset_index(drop=True); #print(df_TCP_latest_packets.shape)
                df_TCP_latest_packets.drop(columns=['Base Packet ID'], inplace=True)   # Drop the helper column
                tot_pkt_reinserted_at_src = df.shape[0] - df_TCP_latest_packets.shape[0]
                '''End'''

                # If a packet is reinserted at source (TCP protocol), consider total packets generated and received for the last most entry
                df_TCP_src = pd.read_csv(os.path.join(root, df['Pkt ID'][0].split('_')[1]+'.csv'))
                df_TCP_src['Base Packet ID'] = df_TCP_src['Packet ID'].apply(get_base_packet_id); #print(df_TCP.shape)
                
                # Group by the base packet ID and get the last entry for each group
                df_TCP_src_latest_packets = df_TCP_src.groupby('Base Packet ID').tail(1).reset_index(drop=True); #print(df_TCP_latest_packets.shape)
                df_TCP_src_latest_packets.drop(columns=['Base Packet ID'], inplace=True)   # Drop the helper column

                total_generated_pkts = len([packet for packet in df_TCP_src_latest_packets['Packet ID'] if file.startswith('Flow_' + packet.split('_')[1])])
                
                # Print rows where Delay (in s) is either zero or infinity
                invalid_delay_df = df_TCP_latest_packets[(df_TCP_latest_packets['Delay (in s)'] == 0) | (df_TCP_latest_packets['Delay (in s)'] == float('inf'))]
                if not invalid_delay_df.empty:
                    print("Rows with invalid delay (0 or inf):")
                    print(invalid_delay_df)
                
                # Filter rows where Delay (in s) is neither zero nor infinity
                filtered_df = df_TCP_latest_packets[(df_TCP_latest_packets['Delay (in s)'] != 0) & (df_TCP_latest_packets['Delay (in s)'] != float('inf'))]
                total_rcvd_pkts = len(filtered_df)
                avg_delay = filtered_df['Delay (in s)'].sum()/total_rcvd_pkts if total_rcvd_pkts != 0 else 0
                
                # Calculate instantaneous PDR for this flow
                flow_id = file.replace('.csv', '')
                instantaneous_pdr_df = calculate_instantaneous_pdr(df_TCP_latest_packets, time_window=1.0)
                if not instantaneous_pdr_df.empty:
                    instantaneous_pdr_df['Flow_ID'] = flow_id
                    instantaneous_pdr_df['File_Path'] = file_path
                    instantaneous_pdr_data[flow_id].append(instantaneous_pdr_df)
                
                # Get # packet forwards
                total_forwards = 0
                if 'Times Packet Forwarded' in df.columns:
                    total_forwards = df['Times Packet Forwarded'].sum()
                
                # Get # retarnsmissions
                total_retransmissions = 0
                if 'Retransmissions' in df.columns:
                    total_retransmissions = df['Retransmissions'].sum()

                # print(file_path, "Packets received:", total_rcvd_pkts, "Avg delay:", avg_delay, "Retransmissions:", total_retransmissions, "Packet forwards:", total_forwards, "\n")
                def extract_last_run_number(filename):
                    """Extracts the last run number from a filename with the pattern ...Flows_Run<No>."""
                    match = re.search(r'Flows_Run(\d+)$', filename)
                    return int(match.group(1)) if match else float('inf')
                # runNo = extract_last_run_number(root)
                # SD:

                # Get current working directory
                current_dir = os.getcwd()
                dirname = os.path.basename(current_dir)
                # Extract number from 'WAR1', 'WAR2', etc.
                match = re.match(r'WAR(\d+)', dirname)
                if match:
                    runNo = int(match.group(1))
                    print("dirname-",dirname, " runNo:", runNo, file=sys.stdout)
                else:
                    runNo = 1
                    print("Parent directory is not in 'WARx' format.")
                
                # Get route switches, uptime and average route length, and packets dropped due to different reasons
                routes = []; route_switches = 0
                route_uptime = 0; hq_route_uptime = 0; tot_sim_time = 0
                var_route_length = 0
                tot_pkt_drop_expiry = 0; tot_pkt_drop_retry = 0; tot_pkt_drop_overflow = 0; tot_pkt_drop_q_mngmnt = 0; tot_pkt_drop_no_route = 0
                try:
                    with open(os.path.join(root,f"run.out"), 'r') as logFile:
                        lines = logFile.readlines()
                    
                    # Get route number for this flow source and destination pair
                    try:
                        # dict_part = ((lines[10]).replace('\n','')).split("Route initialization:  ")[1]
                        for line in lines:
                            if "Route initialization:" in line:
                                dict_part = line.split("Route initialization:")[1].strip()
                                break
                    except:
                        dict_part = ((lines[11]).replace('\n',''))[1:]
                        print("TEST dict_part -------------expect")
                    # print(dict_part)
                    route_dict = ast.literal_eval(dict_part)
                    for routeNo, route in route_dict.items():
                        pair = re.findall(r'\d+', file)
                        print(f"DEBUG Flow file={file}, Extracted src={pair[0]}, dst={pair[1]}")
                        print(f"DEBUG Route dict candidate={route['Route']}")
                        
                        if ((route['Route'][0] == int(pair[0])) and (route['Route'][1]==int(pair[1]))):
                            break
                    # print("routeNo:", routeNo)                   
                    
                    for lineNo in range(10,len(lines)):
                        pattern = re.compile(r"Route details: ({.*})")
                        match = pattern.search(lines[lineNo])
                        if match:
                            # print(lineNo, lines[lineNo])
                            route_details_str = match.group(1)
                            route_details_str = route_details_str.replace('np.True_', 'True').replace('np.False_', 'False') # SD:
                            route_details = ast.literal_eval(route_details_str)
                            tot_sim_time += 1

                            for key, value in route_details.items():
                                if key == routeNo:
                                    curr_route = value['Route']
                                    if ('ValidRoute' in value):
                                        if bool(value['ValidRoute']): 
                                            route_uptime += 1
                                            var_route_length += len(curr_route) -1
                                    if ('HQRoute' in value):
                                        if (bool(value['HQRoute'])): hq_route_uptime += 1
                                    routes.append(curr_route)
                                    # print("Curr route:", curr_route, "is valid:", bool(value['ValidRoute']), "route uptime:", route_uptime, 'HQ route uptime:', hq_route_uptime, 'tot_sim time:', tot_sim_time, "var route length:", var_route_length)
                    #print(routes)
                    # Get route switches
                    for eachTime in range(len(routes)-1):
                        if (routes[eachTime] != routes[eachTime+1]): 
                            route_switches += 1
                    #print("Total route switches: ", route_switches)
                    

                    # Nodes used during simulation
                    nodes_used_for_this_flow = set()
                    for eachRoute in routes:
                        nodes_used_for_this_flow.update(eachRoute)
                    nodes_used_for_this_flow = list(nodes_used_for_this_flow)
                    print("Unique nodes participated for this flow:",nodes_used_for_this_flow)
                    
                    timeout_drop_data: dict[str, list[int]] = {}
                    for eachParticipatingNode in nodes_used_for_this_flow:
                        [drop_expiry, drop_q_mngmnt, drop_retry, drop_overflow, drop_no_route] = get_pkt_drop_count(root, eachParticipatingNode,route['Route'][0], route['Route'][1])
                        tot_pkt_drop_expiry += drop_expiry
                        tot_pkt_drop_q_mngmnt += drop_q_mngmnt
                        tot_pkt_drop_retry += drop_retry
                        tot_pkt_drop_overflow += drop_overflow
                        tot_pkt_drop_no_route += drop_no_route
                        print("node", eachParticipatingNode, "total drops:",drop_expiry+drop_q_mngmnt+drop_retry+drop_overflow+drop_no_route,"drop_expiry",drop_expiry, tot_pkt_drop_expiry, "drop_q_mngmnt",drop_q_mngmnt, tot_pkt_drop_q_mngmnt, "drop_retry",drop_retry, tot_pkt_drop_retry, "drop_overflow",drop_overflow, tot_pkt_drop_overflow, "drop_no_route",drop_no_route, tot_pkt_drop_no_route)
                        if not (f"Node{eachParticipatingNode}" in timeout_drop_data): timeout_drop_data[f"Node{eachParticipatingNode}"] = [0, 0, 0, 0, 0]
                        drop_values = [drop_expiry, drop_q_mngmnt, drop_retry, drop_overflow, drop_no_route]
                        for i, val in enumerate(drop_values):
                            timeout_drop_data[f"Node{eachParticipatingNode}"][i] += val
                    
                    timeout_csv = parent_dir / "timeout_stats.csv"
                    
                    drop_columns = ["drop_expiry", "drop_q_mngmnt", "drop_retry", "drop_overflow", "drop_no_route"]

                    with open(timeout_csv, mode="r", newline="") as csvfile:
                        reader = csv.DictReader(csvfile)
                        headers = reader.fieldnames
                        assert headers is not None
                        headers = list(headers)
                        rows = [row for row in reader]

                    for col in drop_columns:
                        if col not in headers:
                            headers.append(col)

                    csv_dict = {row["Node Name"]: row for row in rows}
                    
                    for node_name, drops in timeout_drop_data.items():
                        if node_name not in csv_dict:
                            csv_dict[node_name] = {h: "" for h in headers}
                            csv_dict[node_name]["Node Name"] = node_name
                        for col, value in zip(drop_columns, drops):
                            existing = int(csv_dict[node_name].get(col, 0) or 0)
                            csv_dict[node_name][col] = existing + value
                    
                    with open(timeout_csv, mode="w", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(csv_dict.values())
                except Exception as e:
                    print("Remove newline from Line 142 in main.py file! Error details:", str(e))
                    print(f"Error occurred: {e.__class__.__name__} - {str(e)}")

                flow_id = file.replace('.csv', '')
                
                new_entry = {'Flow_ID': flow_id,
                            'File Path': file_path,
                            'Run No': runNo,
                            'Packets Generated':total_generated_pkts,
                            'Packets Received':total_rcvd_pkts,
                            'PDR':total_rcvd_pkts/max(1,total_generated_pkts),
                            'Avg Delay': avg_delay,
                            "#Retransmissions": total_retransmissions,
                            '#Forwards': total_forwards,
                            'Avg Route Length': var_route_length/max(1,route_uptime),
                            '%Route Uptime': route_uptime/max(1,tot_sim_time),
                            '%HQ Route Uptime': hq_route_uptime/max(1,tot_sim_time),
                            'Route Switches': route_switches,
                            'Packet Drop (Expiry)':tot_pkt_drop_expiry,
                            'Packet Drop (Queue Management)':tot_pkt_drop_q_mngmnt,
                            'Packet Drop (Retry)':tot_pkt_drop_retry,
                            'Packet Drop (Overflow)':tot_pkt_drop_overflow,
                            'Packet Drop (No Route)':tot_pkt_drop_no_route,
                            'Packet Reinserted (TCP)':tot_pkt_reinserted_at_src}
                records = pd.concat([records, pd.DataFrame([new_entry])], ignore_index=True)
                
                total_exp += 1
                exp_total_generated_pkts += total_generated_pkts
                exp_total_rcvd_pkts += total_rcvd_pkts
                exp_pdr += total_rcvd_pkts/max(1,total_generated_pkts)
                exp_avg_delay += avg_delay
                exp_total_retransmissions += total_retransmissions
                exp_total_forwards += total_forwards
                exp_route_length += var_route_length/max(1,route_uptime)
                exp_route_uptime += route_uptime/max(1,tot_sim_time)
                exp_hq_route_uptime += hq_route_uptime/max(1,tot_sim_time)
                exp_route_switches += route_switches
                
    if total_exp > 0:
        
        # 95% confidence interval and median
        columns_of_interest = ['Packets Generated', 'Packets Received', 'PDR', 'Avg Delay', '#Retransmissions', '#Forwards', 'Avg Route Length', '%Route Uptime', '%HQ Route Uptime' ,'Route Switches']

        # Function to compute statistics
        def compute_statistics(column_data):
            mean = column_data.mean()
            median = column_data.median()
            stderr = column_data.sem()
            confidence_interval = stats.t.interval(0.95, len(column_data) - 1, mean, stderr)
            return mean, median, confidence_interval

        # Dictionary to store the computed statistics
        statistics = {column: compute_statistics(records[column]) for column in columns_of_interest}

        # Creating rows for the statistics
        rows = []
        for stat_name in ['Mean', 'Median', '95% CI Lower', '95% CI Upper']:
            row = ['' for _ in range(len(records.columns))]
            row[0] = stat_name
            rows.append(row)

        # Filling the values for the computed statistics
        for i, column in enumerate(columns_of_interest, start=2):
            mean, median, confidence_interval = statistics[column]
            rows[0][i] = mean
            rows[1][i] = median
            rows[2][i] = confidence_interval[0]
            rows[3][i] = confidence_interval[1]

        # Converting rows to DataFrame and appending to the original DataFrame
        statistics_df = pd.DataFrame(rows, columns=records.columns)
        records.sort_values(by='Run No', ascending=True, inplace=True)
        records = pd.concat([records, statistics_df], ignore_index=True)
        print(records.to_string(index=False))
        records.to_csv(os.path.join(folder_path, 'all_stats.csv'), index=False)
        
        # Save instantaneous PDR data
        if instantaneous_pdr_data:
            all_instantaneous_pdr = []
            for flow_id, pdr_list in instantaneous_pdr_data.items():
                for pdr_df in pdr_list:
                    all_instantaneous_pdr.append(pdr_df)
            
            if all_instantaneous_pdr:
                combined_instantaneous_pdr = pd.concat(all_instantaneous_pdr, ignore_index=True)
                combined_instantaneous_pdr.to_csv(os.path.join(folder_path, 'instantaneous_pdr.csv'), index=False)
                print(f"Instantaneous PDR data saved to {os.path.join(folder_path, 'instantaneous_pdr.csv')}")
                
                print("\n=== Flow-level Summary (All Metrics) ===")
                for flow_id in records['Flow_ID'].dropna().unique():
                    if not flow_id:
                        continue

                    flow_records = records[records['Flow_ID'] == flow_id]
                    if flow_records.empty:
                        continue

                    # Compute averages across runs
                    avg_generated = flow_records['Packets Generated'].mean()
                    avg_received = flow_records['Packets Received'].mean()
                    avg_pdr = flow_records['PDR'].mean()
                    avg_delay = flow_records['Avg Delay'].mean()
                    avg_retrans = flow_records['#Retransmissions'].mean()
                    avg_forwards = flow_records['#Forwards'].mean()
                    avg_expiry = flow_records['Packet Drop (Expiry)'].mean()
                    avg_qm = flow_records['Packet Drop (Queue Management)'].mean()
                    avg_retry = flow_records['Packet Drop (Retry)'].mean()
                    avg_overflow = flow_records['Packet Drop (Overflow)'].mean()
                    avg_no_route = flow_records['Packet Drop (No Route)'].mean()
                    avg_reinserted = flow_records['Packet Reinserted (TCP)'].mean()

                    max_delay = 0
                    perc95_delay = 0
                    for file_path in flow_records['File Path']:
                        df = pd.read_csv(file_path)

                        # Only keep last packet per TCP base ID
                        df['Base Packet ID'] = df['Pkt ID'].apply(lambda x: re.match(r"(.+)_\w+$", x).group(1))
                        df_latest = df.groupby('Base Packet ID').tail(1).reset_index(drop=True)

                        # Filter valid delays
                        df_valid = df_latest[(df_latest['Delay (in s)'] != 0) & (df_latest['Delay (in s)'] != float('inf'))]
                        if df_valid.empty:
                            continue

                        # Absolute max delay
                        max_delay = max(max_delay, df_valid['Delay (in s)'].max())
                        print(df_valid[df_valid['Delay (in s)'] > 5])

                        # 95th percentile delay
                        perc95_delay = max(perc95_delay, np.percentile(df_valid['Delay (in s)'], 95))

                    print(f"\nFlow {flow_id}:")
                    print(f"  Packets Generated: {avg_generated:.2f}")
                    print(f"  Packets Received : {avg_received:.2f}")
                    print(f"  PDR              : {avg_pdr:.3f}")
                    print(f"  Avg Delay (s)    : {avg_delay:.4f}")
                    print(f"  Max Delay (s)    : {max_delay:.4f}")
                    print(f"  95th %ile Delay(s): {perc95_delay:.4f}")
                    print(f"  Retransmissions  : {avg_retrans:.2f}")
                    print(f"  Forwards         : {avg_forwards:.2f}")
                    print(f"  Drops (Expiry)   : {avg_expiry:.0f}")
                    print(f"  Drops (Queue Mgmt): {avg_qm:.0f}")
                    print(f"  Drops (Retry)    : {avg_retry:.0f}")
                    print(f"  Drops (Overflow) : {avg_overflow:.0f}")
                    print(f"  Drops (No Route) : {avg_no_route:.0f}")
                    print(f"  TCP Reinsertions : {avg_reinserted:.0f}")
        
        print("Avg: ", "Packets generated:", round(exp_total_generated_pkts/total_exp), "Packets received:", round(exp_total_rcvd_pkts/total_exp), "PDR:", exp_pdr/total_exp,\
              "Avg delay:", exp_avg_delay/total_exp, "Retransmissions:", round(exp_total_retransmissions/total_exp), "Packet forwards:", round(exp_total_forwards/total_exp),\
              "Route length:", exp_route_length/total_exp, "Route uptime:", exp_route_uptime/total_exp, "HQ Route Uptime:", exp_hq_route_uptime/total_exp, "Route switches:", exp_route_switches/total_exp, "\n" )            

def plot_instantaneous_pdr(csv_path):
    """
    Plot instantaneous PDR vs time for all flows overlaid in a single plot from the CSV file.
    Excludes the last 3 seconds from the time axis and saves the plot to a file.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No instantaneous PDR data to plot.")
        return
    
    flows = df['Flow_ID'].unique()
    
    # Find the maximum time across all flows to exclude last 3 seconds
    max_time = 0
    for flow in flows:
        flow_df = df[df['Flow_ID'] == flow]
        time_starts = flow_df['Time_Window'].apply(lambda x: float(x.split('-')[0]))
        max_time = max(max_time, time_starts.max())
    
    # Exclude last 3 seconds
    cutoff_time = max_time - 4.0
    
    for flow in flows:
        flow_df = df[df['Flow_ID'] == flow]
        # Parse the start of each time window for x-axis
        time_starts = flow_df['Time_Window'].apply(lambda x: float(x.split('-')[0]))
        
        # Filter out data points beyond cutoff_time
        mask = time_starts <= cutoff_time
        filtered_time_starts = time_starts[mask]
        filtered_pdr = flow_df['Instantaneous_PDR'][mask]
        
        plt.plot(filtered_time_starts, filtered_pdr, marker='o', label=flow)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Instantaneous PDR')
    plt.title('Instantaneous PDR vs Time (All Flows) - Excluding Last 3s')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to a file
    plot_filename = os.path.join(os.path.dirname(csv_path), 'instantaneous_pdr_plot.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    
    # Also try to display if possible
    try:
        plt.show()
    except:
        print("Display not available, plot saved to file only.")
    
    plt.close()  # Close the plot to free memory
    

def plot_delay_over_time(records_folder, nth=5, scatter_alpha=0.2, window=100):
    """
    Plot per-packet delay over time for all flows.
    Shows raw delays as scatter + smoothed rolling average as line.
    Automatically adapts to simulation time range.
    """
    delay_plot_path = os.path.join(records_folder, "delay_over_time.png")

    csv_files = [f for f in os.listdir(records_folder) if f.endswith(".csv") and "Flow_Node" in f]
    if not csv_files:
        print("No flow CSV files found for delay plotting.")
        return

    plt.figure(figsize=(10,6))

    for file in csv_files:
        file_path = os.path.join(records_folder, file)
        df = pd.read_csv(file_path)

        # Clean packet IDs
        def get_base_packet_id(packet_id):
            return re.match(r"(.+)_\w+$", packet_id).group(1)
        df['Base Packet ID'] = df['Pkt ID'].apply(get_base_packet_id)
        df = df.groupby('Base Packet ID').tail(1).reset_index(drop=True)

        # Keep only valid delays
        df = df[(df['Delay (in s)'] > 0) & (df['Delay (in s)'] != float('inf'))]
        if df.empty:
            continue

        # Reception time = generation + delay
        df['Reception Time (s)'] = df['Generation Time (in s)'] + df['Delay (in s)']
        df = df.sort_values('Reception Time (s)')

        flow_id = file.replace(".csv", "")

        # Scatter plot (raw points)
        scatter_sample = df.iloc[::nth, :]
        plt.scatter(scatter_sample['Reception Time (s)'],
                    scatter_sample['Delay (in s)'],
                    s=8, alpha=scatter_alpha, label=f"{flow_id} (raw)")

        # Smoothed line (rolling mean)
        df['Delay MA'] = df['Delay (in s)'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['Reception Time (s)'], df['Delay MA'],
                 linewidth=2.5, label=f"{flow_id} (avg)")

    plt.xlabel("Time (s)")
    plt.ylabel("Delay (ms)")
    plt.title("Packet Delay over Time (All Flows)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(delay_plot_path, dpi=300, bbox_inches='tight')
    print(f"Delay-over-time plot saved to: {delay_plot_path}")

    try:
        plt.show()
    except:
        print("Display not available, plot saved to file only.")
    plt.close()

if __name__ == "__main__":
    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("./../Results/").resolve()
    sys.stdout = open(folder / "res.out", "w")
    get_stats(folder, "")
    plot_instantaneous_pdr(os.path.join(folder, "instantaneous_pdr.csv"))
    plot_delay_over_time(folder)
    
