import os
import sys
import pickle as pickle
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn import tree
import re
from netaddr import IPAddress
from statistics import mode
import random
import ipaddress

import warnings
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
global priority
np.random.seed(42)

## import and get entries from trained models ##
clf = pd.read_pickle('rf_model_april_22nd_1pm.pkl')

## list the feature names
feature_names = ['Min differential Packet Length', 'Max differential Packet Length', 'IAT min', 'IAT max', 'Packet Length Total']
split_ranges = {feature: [] for feature in feature_names}

## definition of useful functions
## gets all splits and conditions
def get_splits(forest, feature_names):
    data = []
    #generate dataframe with all thresholds and features
    for t in range(len(forest.estimators_)):
        clf = forest[t]
        n_nodes = clf.tree_.node_count
        features  = [feature_names[i] for i in clf.tree_.feature]
        for i in range(0, n_nodes):
            node_id = i
            left_child_id = clf.tree_.children_left[i]
            right_child_id = clf.tree_.children_right[i]
            threshold = clf.tree_.threshold[i]
            feature = features[i]
            if threshold != -2.0:
                data.append([t, node_id, left_child_id,
                             right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["Tree","NodeID","LeftID","RightID","Threshold","Feature"]
    return data

## gets the feature table of each feature from the splits
def get_feature_table(splits_data, feature_name):
    feature_data = splits_data[splits_data["Feature"]==feature_name]
    feature_data = feature_data.sort_values(by="Threshold")
    feature_data = feature_data.reset_index(drop=True)
    ##
    # feature_data["Threshold"] = (feature_data["Threshold"]).astype(int)
    feature_data["Threshold"] = feature_data["Threshold"].astype(int)
    ##
    code_table = pd.DataFrame()
    code_table["Threshold"] = feature_data["Threshold"]
    #create a column for each split in each tree
    for tree_id, node in zip(list(feature_data["Tree"]), list(feature_data["NodeID"])):
        colname = "s"+str(tree_id)+"_"+str(node)
        code_table[colname] = np.where((code_table["Threshold"] <=
                                        feature_data[(feature_data["NodeID"]== node) &
                                                     (feature_data["Tree"]==tree_id)]["Threshold"].values[0]), 0, 1)
    #add a row to represent the values above the largest threshold
    temp = [max(code_table["Threshold"])+1]
    temp.extend(list([1]*(len(code_table.columns)-1)))
    code_table.loc[len(code_table)] = temp
    code_table = code_table.drop_duplicates(subset=['Threshold'])
    code_table = code_table.reset_index(drop=True)
    return code_table

## Get codes and masks
def get_codes_and_masks(clf, feature_names):
    splits = get_order_of_splits(get_splits_per_tree(clf, feature_names), feature_names)
    depth = clf.max_depth
    codes = []
    masks = []

    for branch, coded in zip(list(retrieve_branches(clf)), get_leaf_paths(clf)):
        code = [0]*len(splits)
        mask = [0]*len(splits)

        for index, split in enumerate(splits):
            if split in branch:
                mask[index] = 1

        masks.append(mask)
        codes.append(code)

    masks = pd.DataFrame(masks)
    masks['Mask'] = masks[masks.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
    
    masks = ["0b" + x for x in masks['Mask']]
    
    indices = range(0,len(splits))
    temp = pd.DataFrame(columns=["split", "index"],dtype=object)
    temp["split"] = splits
    temp["index"] = indices

    final_codes = []

    for branch, code, coded in zip(list(retrieve_branches(clf)), codes, get_leaf_paths(clf)):
        indices_to_use = temp[temp["split"].isin(branch)].sort_values(by="split")["index"]
        
        for i, j in zip(range(0,len(coded)), list(indices_to_use)):
            code[j] = coded[i]

        final_codes.append(code)

    final_codes = pd.DataFrame(final_codes)
    final_codes["Code"] = final_codes[final_codes.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
    
    final_codes = ["0b" + x for x in final_codes["Code"]]
    
    return final_codes, masks

def split_20_bits(code_str):
    if len(code_str) <= 20:
        return [code_str]
    
    segments = [code_str[i:i + 20] for i in range(0, len(code_str), 20)]
    return segments

## get feature tables with ranges and codes only
def get_feature_codes_with_ranges(feature_table, num_of_trees):
    Codes = pd.DataFrame()
    for tree_id in range(num_of_trees):
        colname = "code"+str(tree_id)
        Codes[colname] = feature_table[feature_table[[col for col in feature_table.columns if ('s'+str(tree_id)+'_') in col]].columns[0:]].apply(lambda x: ''.join(x.dropna().astype(str)),axis=1)
    feature_table["Range"] = [0]*len(feature_table)
    feature_table["Range"].loc[0] = "0,"+str(feature_table["Threshold"].loc[0])
    for i in range(1, len(feature_table)):
        if (i==(len(feature_table))-1):
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i])+","+str(feature_table["Threshold"].loc[i])
        else:
            feature_table["Range"].loc[i] = str(feature_table["Threshold"].loc[i-1]+1) + ","+str(feature_table["Threshold"].loc[i])
    Ranges = feature_table["Range"]
    return Ranges, Codes

## get list of splits crossed to get to leaves
def retrieve_branches(estimator):
    number_nodes = estimator.tree_.node_count
    children_left_list = estimator.tree_.children_left
    children_right_list = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]
    # Store the branches paths
    paths = []
    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]
            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output
        else:
            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]
            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])
            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])

## get classes and certainties
def get_classes(clf):
    leaves = []
    classes = []
    certainties = []
    for branch in list(retrieve_branches(clf)):
        leaves.append(branch[-1])
    for leaf in leaves:
        if clf.tree_.n_outputs == 1:
            value = clf.tree_.value[leaf][0]
        else:
            value = clf.tree_.value[leaf].T[0]
        class_name = np.argmax(value)
        certainty = int(round(max(value)/sum(value),2)*100)
        classes.append(class_name)
        certainties.append(certainty)
    return classes, certainties

## get the codes corresponging to the branches followed
def get_leaf_paths(clf):
    depth = clf.max_depth
    branch_codes = []
    for branch in list(retrieve_branches(clf)):
        code = [0]*len(branch)
        for i in range(1, len(branch)):
            if (branch[i]==clf.tree_.children_left[branch[i-1]]):
                code[i] = 0
            elif (branch[i]==clf.tree_.children_right[branch[i-1]]):
                code[i] = 1
        branch_codes.append(list(code[1:]))
    return branch_codes

## get the order of the splits to enable code generation
def get_order_of_splits(data, feature_names):
    splits_order = []
    for feature_name in feature_names:
        feature_data = data[data.iloc[:,4]==feature_name]
        feature_data = feature_data.sort_values(by="Threshold")
        for node in list(feature_data.iloc[:,0]):
            splits_order.append(node)
    return splits_order

def get_splits_per_tree(clf, feature_names):
    data = []
    n_nodes = clf.tree_.node_count
    #set feature names
    features  = [feature_names[i] for i in clf.tree_.feature]
    #generate dataframe with all thresholds and features
    for i in range(0,n_nodes):
        node_id = i
        left_child_id = clf.tree_.children_left[i]
        right_child_id = clf.tree_.children_right[i]
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:
            data.append([node_id, left_child_id,
                         right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["NodeID","LeftID","RightID","Threshold","Feature"]
    return data

## End of model manipulation ##

## Range to ternary conversion ## 

def generate_last_exact_value(feature_index, end_value, hi_binary, codes):
    hi_int = int(hi_binary, 2)

    # Only add `(hi, hi)` if it was NOT already included**
    found_hi = any(end == hi_int for _, end, _ in split_ranges[feature_names[feature_index]])

    if not found_hi and int(end_value, 2) == hi_int - 1:
        mask = generate_mask(feature_index, hi_binary)
        split_ranges[feature_names[feature_index]].append(
            (hex(int(hi_binary, 2)), hex(int(mask, 2)),codes)
        )

def generate_end_value(modified_lo_binary):
    value_binary = '' 
    for i in range(len(modified_lo_binary)):
        if modified_lo_binary[i] == 'x':
            # If 'x' is found, replace it with '0' in value_binary
            value_binary += '1'
        else:
            # Retain '1' in mask_binary
            value_binary += modified_lo_binary[i]
        
    # f.write(f"value_binary: {int(value_binary, 2)}\n")
    return value_binary

def generate_start_value(modified_lo_binary):
    value_binary = '' 
    for i in range(len(modified_lo_binary)):
        if modified_lo_binary[i] == 'x':
            # If 'x' is found, replace it with '0' in value_binary
            value_binary += '0'
        else:
            # Retain '1' in mask_binary
            value_binary += modified_lo_binary[i]
        
    # f.write(f"value_binary: {int(value_binary, 2)}\n")
    return value_binary

def generate_mask(feature_index, modified_binary):
    mask_binary = ''

    # Iterate through each bit in modified_lo_binary
    for i in range(len(modified_binary)):
        if modified_binary[i] == 'x':
            # If 'x' is found, replace it with '0' in mask_binary
            mask_binary += '0'
        elif modified_binary[i] != 'x':
            # If 'x' is not found, replace it with '1' in mask_binary
            mask_binary += '1'

    # Adjust the length of mask_binary based on the feature name
    if feature_names[feature_index] in [
        "IAT min", 
        "Min differential Packet Length",
        "Max differential Packet Length"
        "IAT max"
    ]:
        # Ensure mask_binary is 64 bits long, padded with '1's
        mask_binary = '1' * (32 - len(mask_binary)) + mask_binary
    else:
        # Ensure mask_binary is 16 bits long, padded with '1's
        mask_binary = '1' * (16 - len(mask_binary)) + mask_binary

    # f.write(f"mask_binary: {mask_binary}\n")
    return mask_binary

def handle_trailing_zeros(feature_index, trailing_zeros, lo_binary, hi_binary,codes):
    # Convert lo_binary to a list for mutability
    lo_binary_list = list(lo_binary)

    # Calculate the position where modification should start

    retain_length = len(lo_binary) - trailing_zeros

    # Step 1: Retain bits from MSB till retain_length
    # All bits after retain_length should be set to '0'
    # Example: numbers = [0, 1, 2, 3], numbers[2:] = [20, 30], output = [0, 1, 20, 30]
    lo_binary_list[retain_length:] = ['x'] * trailing_zeros

    # Convert back to a string
    modified_lo_binary = ''.join(lo_binary_list)

    mask = generate_mask(feature_index, modified_lo_binary)
    start_value = generate_start_value(modified_lo_binary)
    end_value = generate_end_value(modified_lo_binary)  

    split_ranges[feature_names[feature_index]].append(
        (hex(int(start_value, 2)), hex(int(mask, 2)),codes)
    )

    generate_last_exact_value(feature_index, end_value, hi_binary,codes) 

def lo_binary_ranges(feature_index, lo_binary, hi_binary,codes):
    lo_binary_list = list(lo_binary)
    
    # If lo + 1 == hi, add directly and return
    if int(lo_binary, 2) + 1 == int(hi_binary, 2):
        if feature_names[feature_index] not in split_ranges:
            split_ranges[feature_names[feature_index]] = []
        
        # Find differing bit positions
        differing_indices = [i for i in range(len(lo_binary)) if lo_binary[i] != hi_binary[i]]

        if len(differing_indices) == len(lo_binary):  # All bits differ
            for value in range(lo, hi + 1):
                mask = generate_mask(feature_index, hi_binary)
                split_ranges[feature_names[feature_index]].append(
                    (hex(int(bin(value)[2:].zfill(len(lo_binary)), 2)), hex(int(mask, 2)),codes)
                )
        else:
            # **Fix: Use modified_binary to preserve `x` bits**
            modified_binary = list(lo_binary)
            modified_binary[-1] = 'x'  # Ensure LSB toggled correctly
            modified_binary = ''.join(modified_binary)

            mask = generate_mask(feature_index, modified_binary)

            split_ranges[feature_names[feature_index]].append(
                (hex(int(lo_binary, 2)), hex(int(mask, 2)),codes)
                # (hex(int(lo_binary, 2)), hex(int(hi_binary, 2)), codes)
            )
        return 
    
    if int(lo_binary, 2) == int(hi_binary, 2):
        if feature_names[feature_index] not in split_ranges:
            split_ranges[feature_names[feature_index]] = []

        mask = generate_mask(feature_index, hi_binary)

        split_ranges[feature_names[feature_index]].append(
            (hex(int(lo_binary, 2)), hex(int(mask, 2)),codes)
            # (hex(int(lo_binary, 2)), hex(int(hi_binary, 2)), codes)
        )
        return 
    
    # Find trailing zeros before the first '1'
    trailing_zeros = 0
    trailing_zeros_index = 0

    for index in range(0, len(lo_binary)):
        if hi_binary[index]  > lo_binary[index]:
            trailing_zeros_index = index
            break

    for bit in reversed(lo_binary[trailing_zeros_index + 1:]):
        if bit == '0':
            trailing_zeros += 1
        else:
            break

    # Handle trailing zeros if found
    if trailing_zeros > 0:
        handle_trailing_zeros(feature_index, trailing_zeros, lo_binary, hi_binary,codes)
    
    # Start modification from right to left
    first_one_found = False

    # Start modification from right to left
    if trailing_zeros > 0:
        first_one_found = True

    for index in range(trailing_zeros + 1, len(lo_binary)):
        bit = lo_binary[-index]
        actual_position = index  # Position in reversed order
        # Identify the first '1' from right to left
        if bit == '1' and not first_one_found and int(lo_binary, 2) != int(hi_binary, 2) - 1:
            mask = generate_mask(feature_index, lo_binary)

            split_ranges[feature_names[feature_index]].append(
                (hex(int(lo_binary, 2)), hex(int(mask, 2)),codes)
            )
            first_one_found = True

        # If the first '1' is found and a '0' appears after it
        elif bit == '0' and first_one_found:
            # Step 1: Set the bit at actual_position to '1'
            lo_binary_list[-actual_position] = '1'

            # Step 2: Set all bits after actual_position to '0'
            lo_binary_list[-actual_position + 1:] = ['x'] * (actual_position - 1)

            # Convert back to a string
            modified_lo_binary = ''.join(lo_binary_list)    
            end_value = generate_end_value(modified_lo_binary)

            if int(end_value, 2) < int(hi_binary, 2):
                mask = generate_mask(feature_index, modified_lo_binary)
                start_value = generate_start_value(modified_lo_binary)
                end_value = generate_end_value(modified_lo_binary)

                # Update split_ranges with the modified value
                split_ranges[feature_names[feature_index]].append(
                    (hex(int(start_value, 2)), hex(int(mask, 2)),codes)
                )

                generate_last_exact_value(feature_index, end_value, hi_binary,codes)

            else:
                return index

def hi_binary_ranges(index, feature_index, hi_binary, lo_binary, lo, hi, codes):
    # Convert hi_binary to a list for mutability
    original_hi_binary_list = list(hi_binary)

    start_index = len(hi_binary) - index + 1

    if index == 0:
        start_index = 1

    if feature_names[feature_index] not in split_ranges:
        split_ranges[feature_names[feature_index]] = []

    last_processed_end = lo - 1  # Track last processed range

    # Start from the second bit, excluding the MSB
    for idx in range(start_index, len(original_hi_binary_list)):
        # Reset to original state at each iteration
        hi_binary_list = original_hi_binary_list[:]

        # If a '1' is found, set it to '0' and convert all following bits to '0'
        if hi_binary_list[idx] == '1':
            hi_binary_list[idx] = '0'

            if idx == len(original_hi_binary_list) - 1 and original_hi_binary_list[-1] == '1':
                hi_binary_list[-1] = 'x'  # Replace LSB with 'x'
            else:
                hi_binary_list[idx+1:] = ['x'] * (len(hi_binary_list) - idx - 1)

            # Convert back to string after each modification
            modified_hi_binary = ''.join(hi_binary_list)

            # Generate start and end values
            mask = generate_mask(feature_index, modified_hi_binary)
            start_value = generate_start_value(modified_hi_binary)
            end_value = generate_end_value(modified_hi_binary)

            # Only include values within the defined range
            if lo <= int(end_value, 2) <= hi:
                split_ranges[feature_names[feature_index]].append(
                    (hex(int(start_value,2)), hex(int(mask, 2)),codes)
                )
                last_processed_end = int(end_value, 2)  # Update last processed range

    # **Fix: Ensure `(hi, hi)` is explicitly included if missing**
    if last_processed_end < hi:
        mask = generate_mask(feature_index, hi_binary)

        split_ranges[feature_names[feature_index]].append(
            (hex(int(hi_binary, 2)), hex(int(mask, 2)),codes)
        )

def generate_ternary_ranges(lo, hi, i,codes):
    # split_ranges[feature_names[i]] = []
    lo_binary = bin(lo)[2:]
    hi_binary = bin(hi)[2:]
    max_length = max(len(lo_binary), len(hi_binary))
    lo_binary = lo_binary.zfill(max_length)
    hi_binary = hi_binary.zfill(max_length)

    get_hi_binary_start_index = lo_binary_ranges(i, lo_binary, hi_binary,codes)
    
    if lo != hi and lo + 1 != hi:
        if get_hi_binary_start_index is not None:
            hi_binary_ranges(get_hi_binary_start_index, i, hi_binary, lo_binary,  lo, hi,codes)
        else:
            hi_binary_ranges(0, i, hi_binary, lo_binary,  lo, hi,codes)
        
## End of Range to ternary conversion ## 

# Get entries for feature tables
tree_code0 = []
tree_code1 = []
tree_code2 = []
tree_code3 = []
tree_code4 = []
tree_code5 = []

for fea in range(0,len(feature_names)):
    # Get table entries and generate file with table entries
    Ranges, Codes = get_feature_codes_with_ranges(get_feature_table(get_splits(clf, feature_names), feature_names[fea]), len(clf.estimators_))

    column_names = Codes.columns.tolist()  # Extract column names

    # print(f"Debug: Codes DataFrame for {feature_names[fea]}:\n", Codes.head())
    for ran, *code_segments in zip(Ranges, Codes.itertuples(index=False, name=None)):
        if(ran == Ranges[len(Ranges)-1]):
            if(feature_names[fea] == "Min differential Packet Length"):
                lo = int(str(ran.split(",")[0]))
                hi = 4294967295  
            elif(feature_names[fea] == "Max differential Packet Length"):
                lo = int(str(ran.split(",")[0]))
                hi = 4294967295  
            elif(feature_names[fea] == "IAT min"):
                lo = int(str(ran.split(",")[0]))
                hi = 4294967295  
            elif(feature_names[fea] == "IAT max"):
                lo = int(str(ran.split(",")[0]))
                hi = 4294967295  
            else:
                lo = int(str(ran.split(",")[0]))
                hi = 65535 
        else:
            lo = int(str(ran.split(",")[0]))
            hi = int(str(ran.split(",")[1]))  

        code_segments_list = [item for sublist in code_segments for item in sublist]

        # Combine column names with their respective values
        column_value_pairs = list(zip(column_names, code_segments_list))
        
        # print(f"Debug: lo={lo}, hi={hi}, fea={fea}, codes={column_value_pairs}")

        generate_ternary_ranges(lo, hi, fea, column_value_pairs)
    
    tree_code_sizes = []

    num_code_segments = len(Codes.columns)  # Total number of segmented columns
    num_trees = len(feature_names)  # Number of trees

    for tree_id in range(num_trees):
        segment_lengths = [len(Codes.iloc[:, col_idx]) - 2 for col_idx in range(tree_id, num_code_segments, num_trees)]
        tree_code_sizes.append(segment_lengths)

    # print(tree_code_sizes, "----------------------------------")

for fea in range(0,len(feature_names)):
    print(feature_names[fea])
    with open(f"rules_{feature_names[fea].replace(' ', '_').lower()}.txt", "w") as entries_file:
        priority=0
        for combination in split_ranges[feature_names[fea]]:
            priority += 1
            value, mask, codes = combination
            values = [f"{value}/{mask}"]
            formatted_values = " ".join(values)

            # print(f"Debug: codes structure for {values} - {codes}")

            # Generate action parameters dynamically
            action_params = []
            for index, action in enumerate(codes):  # Iterate over trees
                param, value = action
                action_params.append(f"{param} {hex(int(value,2))}")
                # action_params.append(f"{param} {value}")

            # Construct the rule string
            rule = (
                f"match {formatted_values} priority {priority} action SetCode{fea} " + " ".join(action_params)
            )

            print(rule, file=entries_file)

for tree_id in range(0, len(clf.estimators_)):
    priority = 0
    with open(f"rules_code_table{str(tree_id)}.txt", "w") as entries_file:
        Final_Codes, Final_Masks = get_codes_and_masks(clf.estimators_[tree_id], feature_names)
        
        Classe, Certain = get_classes(clf.estimators_[tree_id])

        for cod, mas, cla, cer in zip(Final_Codes, Final_Masks, Classe, Certain):
            priority += 1

            # # Pair code and mask segments correctly
            # match_parts = [f"{cod}/{mas}" for cod, mas in zip(cod, mas)]
            # match_str = " ".join(match_parts)  # Join all segments side by side

            # Construct the final rule
            rule = f"match {hex(int(cod,2))}/{hex(int(mas,2))} priority {priority} action SetClass{tree_id} class {cla + 1}"

            print(rule, file=entries_file)  # Write to file

with open(f"rules_voting_table.txt", "w") as entries_file:
    # Get voting table entries
    priority=0
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                try:
                    priority += 1
                    choices = [i, j, k]
                    mode_number = mode(choices)
                    print("match "+  str(i) + " " + str(j) + " " + str(k) + " priority {}".format(priority)+" action set_final_class" + " class_result " + str(mode_number), file=entries_file)
                except:
                    pass


# with open(f"rules_target_flows_table.txt", "w") as entries_file:
#     flow_id_info = pd.read_csv("NIMS_IMA_test_data.csv",usecols=['Flow ID','Label'])
#     flow_id_info = flow_id_info.drop_duplicates(subset=['Flow ID'])
#     for index, flow in flow_id_info.iterrows():
#         flow_id = flow['Flow ID']
#         print(flow_id)
#         id_values = flow_id.split(" ")
#         # With all tuple elements
#         try:
#             print("match "+ str(int(ipaddress.ip_address(id_values[0]))) + \
#                         " " + str(int(ipaddress.ip_address(id_values[1]))) + \
#                         " " + str(id_values[2]) + \
#                         " " + str(id_values[3]) + \
#                         " " + str(id_values[4]) + \
#                         " action set_flow_class class " + str(0), file=entries_file)
#         except:
#             continue

print("** TABLE ENTRIES GENERATED AND STORED IN DESIGNATED FILE **")
