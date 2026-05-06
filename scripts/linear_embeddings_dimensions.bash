#!/usr/bin/env bash

# Activate virtual environment
source ../.venv/bin/activate

OUTPUT_FOLDER="linear_embedding_dimensions"

# PCA components (in specified order)
#PCA_COMPONENTS=(768 512 1024)
#MCCA_COMPONENTS=(64 72 80 88 96 104 112 120 128 136 164 192 256)
#MLP_HIDDEN=(256 512 768 1024)

PCA_COMPONENTS=(768 1024 1280 1512)
MCCA_COMPONENTS=(256 280 300 320 360 384)
MLP_HIDDEN=(128 256 512 768 1024)
DEPTHS=(1 2 3)

#DEPTH=2

for pca in "${PCA_COMPONENTS[@]}"; do
  for mcca in "${MCCA_COMPONENTS[@]}"; do
    echo "Running with PCA=$pca, MCCA=$mcca"

    python 1_within_subject_linear.py \
     --output_folder "$OUTPUT_FOLDER" \
     --n_components_pca "$pca" \
     --n_components_mcca "$mcca"
    
    for d_hidden in "${MLP_HIDDEN[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        echo "Running MLP with hidden dimension $d_hidden and depth $DEPTH"
          python 2b_within_subject_mlp.py \
              --linear_embeddings_dir linear_embedding_dimensions \
              --embeddings_template "ws_linear_v1_${pca}_${mcca}_sub-{subject:02d}.pt" \
              --output_dir "mlp_embeddings_dimensions_depth-${DEPTH}" \
              --version "pca-${pca}_depth-${DEPTH}" \
              --depth "$DEPTH" \
              --d_hidden "$d_hidden"
          python standarize_embeddings.py \
            --input_folder "mlp_embeddings_dimensions_depth-${DEPTH}" \
            --input_template "ws_mlp_pca-${pca}_depth-${DEPTH}_${mcca}_${d_hidden}_sub-{subject:02d}.pt" \
            --output_folder "mlp_embeddings_dimensions_depth-${DEPTH}" \
            --output_template "ws_mlp_pca-${pca}_depth-${DEPTH}_${mcca}-${d_hidden}_sub-{subject:02d}-avg-standarized.pt" \
            --average
            # --input_template "ws_mlp_v1_pca-${pca}_depth-${DEPTH}_hidden-${d_hidden}_sub-{subject:02d}.pt" \
            # --input_template "ws_mlp_v1_pca-${pca}_depth-${DEPTH}_hidden-${d_hidden}_sub-{subject:02d}.pt" \
          done
    done
  done
done


# python 5_mini_vec2vec_alignment.py --input_folder "mlp_embeddings_dimensions_depth-2" --input_template "ws_mlp_pca-1024_depth-2_256-256_sub-{subject:02d}-avg-standarized.pt" --output_folder "mini_vec2vec_dimensions" --output_template "mini_vec2vec_pca-1024_depth-2_d-256_dh-256-sub-{subject_x:02d}-sub-{subject_y:02d}.pt" --subject_x 1 --subject_y 5
