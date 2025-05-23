# Change the output file path based on test or train dataset
for f in ./pcaps/test_data/*.pcap
	do
		echo $f
        tshark -r $f -Y 'ip.proto == 6 or ip.proto == 17' -T fields -e frame.time_relative -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -E separator='|' > $f.txt
	done