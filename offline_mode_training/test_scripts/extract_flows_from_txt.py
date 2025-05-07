import pandas as pd
import numpy as np
import sys
import os

# Read input arguments
filename_in = sys.argv[1]
filename_out = sys.argv[2]
npkts = int(sys.argv[3])

# Load packet data
packet_data = pd.DataFrame()
packet_data = pd.read_csv(filename_in, sep='|', header=None)
packet_data.columns = ['timestamp', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'ip.proto', 'ip.len', 'udp.srcport', 'udp.dstport']

# Filter protocol and drop irrelevant data
packet_data = packet_data[(packet_data["ip.proto"] != "1,17") & (packet_data["ip.proto"] != "1,6")].reset_index(drop=True)
packet_data = packet_data.dropna(subset=['ip.proto'])
packet_data["ip.src"] = packet_data["ip.src"].astype(str)
packet_data["ip.dst"] = packet_data["ip.dst"].astype(str)
packet_data["ip.len"] = packet_data["ip.len"].astype("int")
##
packet_data["tcp.srcport"] = packet_data["tcp.srcport"]
packet_data["tcp.dstport"] = packet_data["tcp.dstport"]
packet_data["udp.srcport"] = packet_data["udp.srcport"].astype('Int64')
packet_data["udp.dstport"] = packet_data["udp.dstport"].astype('Int64')

# Select source and destination port based on protocol
packet_data["srcport"] = np.where(packet_data["ip.proto"] == "6", packet_data["tcp.srcport"], packet_data["udp.srcport"])
packet_data["dstport"] = np.where(packet_data["ip.proto"] == "6", packet_data["tcp.dstport"], packet_data["udp.dstport"])
#
packet_data["srcport"] = np.where(packet_data["ip.proto"] == 6, packet_data["tcp.srcport"], packet_data["udp.srcport"])
packet_data["dstport"] = np.where(packet_data["ip.proto"] == 6, packet_data["tcp.dstport"], packet_data["udp.dstport"])
#
packet_data["srcport"] = packet_data["srcport"].astype('Int64')
packet_data["dstport"] = packet_data["dstport"].astype('Int64')

# Drop unnecessary columns
packet_data = packet_data.drop(["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"], axis=1)
packet_data = packet_data.reset_index(drop=True)

# Create flow IDs
packet_data["flow.id"] = (
    packet_data["ip.src"].astype(str) + " " +
    packet_data["ip.dst"].astype(str) + " " +
    packet_data["srcport"].astype(str) + " " +
    packet_data["dstport"].astype(str) + " " +
    packet_data["ip.proto"].astype(str)
)

# Labeling
filename_patterns = {
                        "discord"         : "discord", 
                        "whatsapp"        : "whatsapp", 
                        "signal"          : "signal", 
                        "telegram"        : "telegram", 
                        "messenger"       : "messenger", 
                        "teams"           : "teams" 
                    }

for pattern, labeld in filename_patterns.items():
    if pattern in filename_in:
        label = labeld

number_of_pkts_limit, min_number_of_packets = npkts, npkts

# Initialize dictionaries for flows and their features
main_packet_size = {}  # Key: flow.id, Value: list of packet lengths
flow_list = []  # List of flow IDs
differential_packet_size = {}  # Key: flow.id, Value: list of differential packet sizes
main_inter_arrival_time = {}  # dictionary to store list of IATs for each flow (Here key = flowID, value = list of IATs)
last_time = {}  # for each flow we store timestamp of the last packet arrival
avg_pkt_sizes = {}  # contains the flowID and their calculated average packet sizes
labels = {}  # contains the flowID and their labels
flow_start_time = {}  # First packet timestamp for each flow
flow_end_time = {}    # Last packet timestamp for each flow

# Collect packets into flows
print("NOW: COLLECTING PACKETS INTO FLOWS...")
for row in packet_data.itertuples(index=True, name='Pandas'):
    time    = float(row[1])    # timestamp of the packet
    srcip   = row[2]          #src ip
    dstip   = row[3]          #dst ip
    pktsize = row[5]        #packet size   
    proto   = row[4]         #protocol
    srcport = row[6]     #source port
    dstport = row[7]     #destination port
    key     = row[8]          #key which is a concatenation of the 5-tuple to identify the flow

    if key in flow_list:
        if len(main_packet_size[key]) < number_of_pkts_limit:
            main_packet_size[key].append(pktsize)  # Append packet size

            # Update flow end time
            flow_end_time[key] = time

            # Calculate differential packet length
            if (pktsize > main_packet_size[key][-2]):
                diff_len = pktsize - main_packet_size[key][-2]  # Difference with previous packet
            else:
                diff_len = main_packet_size[key][-2] - pktsize # Difference with previous packet

            differential_packet_size[key].append(diff_len)
            labels[key] = label

            # Calculate IAT
            lasttime = last_time[key]
            diff = round(float(time) - float(lasttime), 9)  # calculate inter-arrival time (seconds)
            main_inter_arrival_time[key].append(diff)  # append IAT
            last_time[key] = time  # update last time for the flow, to the timestamp of this packet

    else:
        # Initialize new flow
        flow_list.append(key)
        labels[key] = label
        main_packet_size[key] = [pktsize]
        main_inter_arrival_time[key] = []
        differential_packet_size[key] = []
        flow_start_time[key] = time  # Initialize flow start time
        flow_end_time[key] = time  # Initialize flow end time
        last_time[key] = time

# Write output to CSV
print("NOW: WRITING FLOW FEATURES INTO CSV...")
header = "Flow ID, Min Packet Length, Max Packet Length, Packet Length Total, Min differential Packet Length, Max differential Packet Length, IAT min, IAT max, Flow Duration,Label"
file_exists = os.path.isfile(filename_out)
with open(filename_out, "a") as text_file:
    if not file_exists:  # Write header only if the file doesn't exist
        text_file.write(header + "\n")

    for key in flow_list:
        # Convert lists to strings for writing
        packet_list = main_packet_size[key]  # packet_list contains the list of packet sizes for the flow in consideration
        length = len(packet_list)  # number of packets
        total_bytes = sum(packet_list)

        avg_pkt_sizes[key] = sum(packet_list) / length  # calculate avg packet size, and store
        min_pkt_size = min(packet_list)
        max_pkt_size = max(packet_list)
        
        inter_arrival_time_list = main_inter_arrival_time[key]  # a list containing IATs for the flow
        length = len(inter_arrival_time_list)
        if length == 0:
            min_IAT = 0
            max_IAT = 0
        else:
            min_IAT = min(inter_arrival_time_list)
            min_IAT_ms = round(1000000*min_IAT, 9) # convert in nanoseconds
            max_IAT = max(inter_arrival_time_list)
            max_IAT_ms = round(1000000*max_IAT, 9) # convert in nanoseconds
        
        if length > 0:
            flow_duration = sum(inter_arrival_time_list)  # flow duration seconds
            flow_duration_ms = round(1000000000*flow_duration, 9) # convert in nanoseconds

        # pkt_lengths_str = " ".join(map(str, main_packet_size[key]))
        diff_lengths_str = " ".join(map(str, differential_packet_size[key]))
        min_diff_pkt_size = min(differential_packet_size[key]) if differential_packet_size[key] else 0
        max_diff_pkt_size = max(differential_packet_size[key]) if differential_packet_size[key] else 0
        print(f"{key},{str(min_pkt_size)},{str(max_pkt_size)},{str(sum(packet_list))},{str(min_diff_pkt_size)},{str(max_diff_pkt_size)},{str(min_IAT_ms)},{str(max_IAT_ms)},{str(flow_duration_ms)},{str(labels[key])}\n")
        if(len(main_packet_size[key]) >= min_number_of_packets):
            text_file.write(f"{key},{str(min_pkt_size)},{str(max_pkt_size)},{str(sum(packet_list))},{str(min_diff_pkt_size)},{str(max_diff_pkt_size)},{str(min_IAT_ms)},{str(max_IAT_ms)},{str(flow_duration_ms)},{str(labels[key])}\n")