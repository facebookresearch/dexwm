#!/bin/bash

python robocasa-murp/robocasa/scripts/stats.py --root_dir /fsx-siro/achvysh07/projects/mimicgen_data_new_7_30/new_batch_teleop/mimicgen_dist_adjusted_seed_cluster_july29_smooth_batch
python robocasa-murp/robocasa/scripts/combine_demos.py --json_file /fsx-siro/achvysh07/projects/robot-skills-sim/output_stats.json --output combined_mg_test0.hdf5 --mode parallel
python  robocasa-murp/robocasa/scripts/qa.py --h5_file_path /fsx-siro/achvysh07/projects/robot-skills-sim/combined_mg_test0.hdf5 --save_dir /fsx-siro/achvysh07/projects/ --clean_h5_path combined_mg_test_clean.hdf5
