#!/usr/bin/env bash

# Activate virtual environment
source ../.venv/bin/activate

OUTPUT_FOLDER="final_alignment"

N_ITERATIONS=10

for i in $(seq 1 $N_ITERATIONS); do
  echo "Running iteration $i of $N_ITERATIONS"

  python 4_mini_vec2vec_alignment.py --output_folder "final_alignment_${i}" --n_runs_refinement 3000
  # python 5_mini_vec2vec_alignment_models.py  --n_runs_refinement 3000

done

