#!/bin/bash

# use GNU parallel to run multiple repetitions and scenarios in parallel
# run from project root! (where Readme is)
parallel rlsp :::: scripts/agent_config_files.txt :::: scripts/network_files.txt :::: scripts/service_files.txt :::: scripts/config_files.txt ::: "500" ::: "--seed" :::: scripts/10seeds.txt ::: "--append-test"

# select and test best agent 30x (of all results with these inputs! also from previous runs)
printf "\n\nSelecting and testing best agent again\n\n"
parallel rlsp :::: scripts/agent_config_files.txt :::: scripts/network_files.txt :::: scripts/service_files.txt :::: scripts/config_files.txt ::: "1" ::: "--best" ::: "--sim-seed" :::: scripts/30seeds.txt
