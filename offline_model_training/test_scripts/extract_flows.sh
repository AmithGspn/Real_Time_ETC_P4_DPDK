#!/bin/bash

# Change the output file path based on test or train dataset
output_file="NIMS_IMA_test_data.csv"

if [ ! -f $output_file ]; then
    echo "Flow ID, Min Packet Length, Max Packet Length, Packet Length Total, Min differential Packet Length, Max differential Packet Length, IAT min, IAT max, Flow Duration, Label" > $output_file
fi

for f in ./pcaps/test_data/*.txt
    do
        echo "Processing $f"
        python3 extract_flows_from_txt.py $f $output_file 8
    done

echo "Feature extraction completed for all files."
