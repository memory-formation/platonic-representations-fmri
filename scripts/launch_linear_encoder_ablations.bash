


python 1_within_subject_linear_ablations.py --rois 0 --output_folder "ablations" --version "skip_rel_mcca_distill" \
    --skip_reliability \
    --skip_mcca \
    --skip_mcca_distill

python 1_within_subject_linear_ablations.py --rois 0 --output_folder "ablations" --version "skip_rel_mcca" \
    --skip_mcca \
    --skip_mcca_distill

python 1_within_subject_linear_ablations.py --rois 0 --output_folder "ablations" --version "skip_rel" \
    --skip_mcca \

python 1_within_subject_linear_ablations.py --rois 0 --output_folder "ablations" --version "skip_mcca" \
    --skip_reliability \
    --skip_mcca \