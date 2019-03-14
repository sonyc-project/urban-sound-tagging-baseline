#!/usr/bin/env bash

# Extract embeddings
pushd urban-sound-tagging-baseline
python extract_embedding.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH $FEATURES_PATH $VGGISH_PATH

# Train fine-level model and produce predictions
python classify.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml $FEATURES_PATH/vggish $OUTPUT_PATH baseline_fine --label_mode fine

# Evaluate model based on AUPRC metric
python evaluate_predictions.py $OUTPUT_PATH/baseline_fine/*/output_mean.csv $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml

# Train coarse-level model and produce predictions
python classify.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml $FEATURES_PATH/vggish $OUTPUT_PATH baseline_coarse --label_mode coarse

# Evaluate model based on AUPRC metrics
python evaluate_predictions.py $OUTPUT_PATH/baseline_coarse/*/output_mean.csv $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml

# Return to the base directory
popd
