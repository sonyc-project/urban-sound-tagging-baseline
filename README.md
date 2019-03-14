# DCASE 2019 Challenge: Task 5 - Urban Sound Tagging

## Instructions for replicating baseline
```shell
# Clone repository
$ git clone https://github.com/sonyc-project/urban-sound-tagging-baseline.git
$ cd urban-sound-tagging-baseline
# Create environment
$ conda create -n ust python=3.6
$ source activate ust
# Install dependencies
$ pip install -r requirements.txt
# Download VGGish model files
$ mkdir vggish_model
$ cd vggish_model
$ curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
$ curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
$ cd ../
# Download dataset
$ mkdir sonyc-ust
$ cd sonyc-ust
$ wget https://zenodo.org/record/2590742/files/annotations.csv
$ wget https://zenodo.org/record/2590742/files/audio.tar.gz
$ wget https://zenodo.org/record/2590742/files/dcase-ust-taxonomy.yaml
$ wget https://zenodo.org/record/2590742/files/README.md
# Decompress audio
$ tar xf audio.tar.gz
$ rm audio.tar.gz
$ cd ../
# Extract embeddings
$ python extract_embedding.py ./sonyc-ust/annotations.csv ./sonyc-ust ./features vggish_model
# Train fine-level model and produce predictions
$ python classify.py ./sonyc-ust/annotations.csv ./sonyc-ust/dcase-ust-taxonomy.yaml ./features/vggish ./output baseline_fine --label_mode fine
# Evaluate model based on AUPRC-like metric
$ python evaluate_predictions.py ./output/baseline_fine/<timestamp>/output_mean.csv ./sonyc-ust/annotations.csv ./sonyc-ust/dcase-ust-taxonomy.yaml
# Train coarse-level model and produce predictions
$ python classify.py ./sonyc-ust/annotations.csv ./sonyc-ust/dcase-ust-taxonomy.yaml ./features/vggish ./output baseline_fine --label_mode coarse
# Evaluate model based on AUPRC-like metric
$ python evaluate_predictions.py ./output/baseline_coarse/<timestamp>/output_mean.csv ./sonyc-ust/annotations.csv ./sonyc-ust/dcase-ust-taxonomy.yaml
```
