#!/bin/bash

base_path="0-20/"

paths=(
    "0-20/alpha_0.005_ply"
    "0-20/alpha_0.005_ply"
    "0-20/alpha_0.01_ply"
    "0-20/alpha_0.02_ply"
    "0-20/alpha_0.03_ply"
    "0-20/alpha_0.06_ply"
    "0-20/alpha_0.09_ply"
    "0-20/alpha_0.12_ply"
    "0-20/alpha_0.15_ply"
    "0-20/alpha_0.18_ply"
    "0-20/alpha_0.21_ply"
)

dir_names=(
    "alpha_0.005_ply"
    "alpha_0.01_ply"
    "alpha_0.02_ply"
    "alpha_0.03_ply"
    "alpha_0.06_ply"
    "alpha_0.09_ply"
    "alpha_0.12_ply"
    "alpha_0.15_ply"
    "alpha_0.18_ply"
    "alpha_0.21_ply"
)

for i in "${!paths[@]}"; do
    path="${paths[$i]}"
    dir_name="${dir_names[$i]}"

    echo "Dir ${dir_name}"

    python3 rotation.py \
        --path "$path" \
        --frame_path "${base_path}/${dir_name}_frame_s_0.4" \
        --zoom 0.4

    python3 rotation.py \
        --path "$path" \
        --frame_path "${base_path}/${dir_name}_frame_s_0.6" \
        --zoom 0.6
done