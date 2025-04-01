#!/bin/bash

frame_dir_06=(
    "../0-20/alpha_0.005_ply_frame_s_0.6"
    "../0-20/alpha_0.01_ply_frame_s_0.6"
    "../0-20/alpha_0.02_ply_frame_s_0.6"
    "../0-20/alpha_0.03_ply_frame_s_0.6"
    "../0-20/alpha_0.06_ply_frame_s_0.6"
    "../0-20/alpha_0.09_ply_frame_s_0.6"
    "../0-20/alpha_0.12_ply_frame_s_0.6"
    "../0-20/alpha_0.15_ply_frame_s_0.6"
    "../0-20/alpha_0.18_ply_frame_s_0.6"
    "../0-20/alpha_0.21_ply_frame_s_0.6"
)

frame_dir_04=(
    "../0-20/alpha_0.005_ply_frame_s_0.4"
    "../0-20/alpha_0.01_ply_frame_s_0.4"
    "../0-20/alpha_0.02_ply_frame_s_0.4"
    "../0-20/alpha_0.03_ply_frame_s_0.4"
    "../0-20/alpha_0.06_ply_frame_s_0.4"
    "../0-20/alpha_0.09_ply_frame_s_0.4"
    "../0-20/alpha_0.12_ply_frame_s_0.4"
    "../0-20/alpha_0.15_ply_frame_s_0.4"
    "../0-20/alpha_0.18_ply_frame_s_0.4"
    "../0-20/alpha_0.21_ply_frame_s_0.4"
)

alpha_values=(
    "alpha_0.005"
    "alpha_0.01"
    "alpha_0.02"
    "alpha_0.03"
    "alpha_0.06"
    "alpha_0.09"
    "alpha_0.12"
    "alpha_0.15"
    "alpha_0.18"
    "alpha_0.21"
)

output_path="0-20_output.json"


for i in "${!frame_dir_06[@]}"; do
    frame_06="${frame_dir_06[$i]}"
    frame_04="${frame_dir_04[$i]}"
    alpha="${alpha_values[$i]}"

    echo "Running with alpha ${alpha}"

    echo "Dir ${alpha}"
    python3 test.py \
        --database REHEARSE \
        --csv_path database/rehearse_data_info/0-20.csv \
        --frame_dir_06 "$frame_06" \
        --frame_dir_04 "$frame_04" \
        --alpha "$alpha" \
        --output_path "$output_path"
done