# DCASE 2019 Challenge: Task 5 - Urban Sound Tagging

## Installation
You'll need [Python 3](https://www.python.org/download/releases/3.0/) and [Anaconda](https://www.anaconda.com/distribution/) installed, and will need a bash terminal environment.

Before doing anything else, clone this repository and enter it:

```shell
git clone https://github.com/sonyc-project/urban-sound-tagging-baseline.git
cd urban-sound-tagging-baseline
```

### Quick Start

To get started quickly, simply run:

```shell
# Replace with your preferred directories:
SONYC_UST_PATH=~/datasets/sonyc-ust # Where dataset will be downloaded
VGGISH_PATH=~/models/vggish # Where VGGish files will be downloaded
./setup.sh
```

### Setup Guide

If you want to go through the motions of setting up the environment, you can follow this guide.

First, set up some environment variables to make things easier for yourself. Feel free to change these to directories that work better for you.

```shell
SONYC_UST_PATH=~/datasets/sonyc-ust # Where dataset will be downloaded
VGGISH_PATH=~/models/vggish # Where VGGish files will be downloaded
```

Then set up your Python environment:

```shell
conda create -n ust python=3.6
source activate ust
pip install -r requirements.txt
```

We're using [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) features as our input representation, so download the required model files:

```shell
mkdir -p $VGGISH_PATH
pushd $VGGISH_PATH
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
popd
```

Now, download the dataset from [Zenodo](https://zenodo.org/record/2590742) and decompress the audio files:
```shell
mkdir -p $SONYC_UST_PATH
pushd $SONYC_UST_PATH
wget https://zenodo.org/record/2590742/files/annotations.csv
wget https://zenodo.org/record/2590742/files/audio.tar.gz
wget https://zenodo.org/record/2590742/files/dcase-ust-taxonomy.yaml
wget https://zenodo.org/record/2590742/files/README.md
tar xf audio.tar.gz
rm audio.tar.gz
popd
```

Your environment is now set up!


## Replicating baseline
### Quick Start

To get started immediately (assuming you've set up your environment), you can just run:

```shell
# Replace with your preferred directories:
SONYC_UST_PATH=~/datasets/sonyc-ust # Dataset location
VGGISH_PATH=~/models/vggish # VGGish model files location
FEATURES_PATH=~/features/sonyc-ust # Where VGGish features will be stored
OUTPUT_PATH=~/output/sonyc-ust # Where model output files will be stored
./baseline_example.sh
```

### Baseline Guide


First, set up some environment variables to make things easier. Feel free to change these to directories that work better for you.

```shell
SONYC_UST_PATH=~/datasets/sonyc-ust # Dataset location
VGGISH_PATH=~/models/vggish # VGGish model files location
FEATURES_PATH=~/features/sonyc-ust # Where VGGish features will be stored
OUTPUT_PATH=~/output/sonyc-ust # Where model output files will be stored
```

Enter the source code directory within the repository:

```shell
cd urban-sound-tagging-baseline
```

Extract embeddings from the SONYC-UST data:

```shell
python extract_embedding.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH $FEATURES_PATH $VGGISH_PATH
```

Now, train a fine-level model and produce predictions:

```shell
python classify.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml $FEATURES_PATH/vggish $OUTPUT_PATH baseline_fine --label_mode fine
```

Evaluate the fine-level model output file (using frame-averaged clip predictions) on AUPRC:

```shell
python evaluate_predictions.py $OUTPUT_PATH/baseline_fine/*/output_mean.csv $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml
```

Now, train a coarse-level model and produce predictions:

```shell
python classify.py $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml $FEATURES_PATH/vggish $OUTPUT_PATH baseline_coarse --label_mode coarse
```

Evaluate the coarse-level model output file (using frame-averaged clip predictions) on AUPRC:

```shell
python evaluate_predictions.py $OUTPUT_PATH/baseline_coarse/*/output_mean.csv $SONYC_UST_PATH/annotations.csv $SONYC_UST_PATH/dcase-ust-taxonomy.yaml
```
