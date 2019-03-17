#!/usr/bin/env bash

# Create environment
conda create -n sonyc-ust python=3.6 -y
source activate sonyc-ust

# Install dependencies
yes | pip install -r requirements.txt

# Download VGGish model files
mkdir -p $SONYC_UST_PATH/vggish
pushd $SONYC_UST_PATH/vggish
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
popd

# Download dataset
mkdir -p $SONYC_UST_PATH/data
pushd $SONYC_UST_PATH/data
wget https://zenodo.org/record/2590742/files/annotations.csv
wget https://zenodo.org/record/2590742/files/audio.tar.gz
wget https://zenodo.org/record/2590742/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/2590742/files/README.md

# Decompress audio
tar xf audio.tar.gz
rm audio.tar.gz
popd

