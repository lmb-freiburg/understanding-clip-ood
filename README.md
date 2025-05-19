# When and How Does CLIP Enable Domain and Compositional Generalization?

Official code for our paper ["When and How Does CLIP Enable Domain and Compositional Generalization?"](https://arxiv.org/abs/2502.09507) (ICML 2025 spotlight).

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{kempf2025and,
    title={When and How Does CLIP Enable Domain and Compositional Generalization?},
    author={Kempf, Elias and Schrodi, Simon and Argus, Max and Brox, Thomas},
    booktitle={Proceedings of the 42nd International Conference on Machine Learning},
    year={2025}
}
```

## Environment Setup

We recommend using Python 3.10 with which this code was developed and tested. After cloning, you can install the package as follows:
```bash
pip install -e .
pip install -e deps/open_clip/
pip install -e deps/sparse_autoencoder/
```

## Data Setup

To reproduce our main experiments, you need the DomainNet and the ImageNet-Captions datasets. Optionally, you can also
use CC3M and CC12M instead of ImageNet-Captions as base dataset.

### DomainNet

Either download the (cleaned) DomainNet dataset from [here](https://ai.bu.edu/DomainNet/) or use the provided download script:
```bash
. data/download_domainnet.sh
```
In either case, the directory containing the dataset needs to be writable since some scripts will attempt to create new
files there. After downloading, generate captions for DomainNet by running the following script (adjust `domainnet_path`
if you used a different location):
```bash
python scripts/generate_domainnet_captions.py --domainnet_path data/domainnet
```

### ImageNet-Captions

Download the `imagenet_captions.zip` file from the official GitHub [repo](https://github.com/mlfoundations/imagenet-captions)
and unpack it to the **_data_** directory. To download the exact file version our work was based on, you can use:
```bash
wget https://github.com/mlfoundations/imagenet-captions/raw/5cf98361f5e67661fd5b2c6ee219567484440da9/imagenet_captions.zip
unzip imagenet_captions.zip
```
Please note that this file only provides the textual data and the names of the corresponding images from the official
ImageNet training dataset. So you also need to download the ImageNet training set (or at least the corresponding subset
of it). Afterwards, you can create the TSV files we used for ImagetNet-Captions training by running:
```bash
python scripts/generate_imagenet_captions.py --imagenet_train_path <path/to/imagenet/train>
```

### Creating Domain Mixtures

After we have downloaded both DomainNet and ImageNet-Captions, we can now re-create the domain mixtures from the paper.
This can be done using the provided SLURM [script](slurm/subsample-domainnet.sh). You can either run this script
directly in your shell or submit it via SLURM after addressing all the TODOs in the script:
```bash
sbatch slurm/subsample-domainnet.sh
```
By default the script only creates the domain mixtures of our main experiments (e.g., Figure 2). To also create the
mixtures for our various interpolation experiments, you can comment out the respective lines in the script. However,
please note that creating all TSV indices will take quite a bit of disk space (~20GB).

The generated TSV indices adhere to the following naming convetion:
```
combined-captions-[split]-lso-[domains]-no[testdomain]classes.tsv
```
where `split` is either train or val, `domains` are the first letters of all included domains (e.g., cipqrs if all six
domains are included), and `testdomain` indicates the domain we want to test on (i.e., from which we excluded the 15
test classes). For example, for sketch as the test domain, we would have the following domain mixtures:
- combined-captions-train-lso-real-only.tsv (Natural-only, only real images, the same for all test domains)
- combined-captions-train-lso-rs-nosketchclasses.tsv (CG low-diversity, only real and sketch domains are included)
- combined-captions-train-lso-cipqrs-nosketchclasses.tsv (CG high-diversity, all domains are included)
- combined-captions-train-lso-cipqr-nosketchclasses.tsv (Leave-out-domain, all domains except sketch are included)


### CC3M / CC12M

If you want to use either of these as a base dataset, please follow the corresponding instructions to download the
datasets ([CC3M](https://github.com/google-research-datasets/conceptual-captions) / [CC12M](https://github.com/google-research-datasets/conceptual-12m)).
Afterwards, you need to create TSV files for the train and validation splits (e.g., _cc3m-train.tsv_ / _cc3m-val.tsv_) 
and put them under **_data/indices_**. These files should have the following format:
```
filepath    title
img_path_1    img_caption_1
img_path_2    img_caption_2
...
```
Finally, you can merge these base datasets with our domain mixtures like this:
```bash
python scripts/merge_ccxm.py --mode cc3m
python scripts/merge_ccxm.py --mode cc12m
```

## Training

### CLIP

We used OpenCLIP and SLURM to train our CLIP models. For example, you can run the natural-only experiment like this:
```bash
cd deps/open_clip
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --train-data "../../data/indices/combined-captions-train-lso-real-only.tsv" \
    --val-data "../../data/indices/combined-captions-val-lso-real-only.tsv" \
    --save-frequency 1 \
    --save-most-recent \
    --report-to tensorboard \
    --lr 0.001 \
    --warmup 500 \
    --batch-size=128 \
    --accum-freq 2 \
    --epochs=32 \
    --workers=6 \
    --model RN50 \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 50 \
    --name "clip/RN50-lso-real-only-s0"
```
Note that we trained our ImageNet-Captions models with an effective batch size of 1024 (i.e., 128 samples per GPU across
4 GPUs and gradient accumulation frequency of 2). Make sure to adjust the `batch-size` and `accum-freq` parameters
accordingly depending on your setup. See this [script](slurm/train-clip.sh) for more details. If you do not want to use
SLURM, you can also run the training using `torchrun`. In this case, please refer to the official
[open_clip](https://github.com/mlfoundations/open_clip) documentation for details. Model checkpoints and logs will be
stored under **_deps/open\_clip/logs_**.

If you want to run the experiments with CC3M or CC12M as the base dataset, you need to adjust the hyperparameters and
the TSV datasets. You can use this [script](slurm/train-clip-ccxm.sh) for reference. For the natural-only example with
CC12M, the command should like something like this:
```bash
cd deps/open_clip
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --train-data "../../data/indices/cc12m-train-lso-real-only.tsv" \
    --val-data "../../data/indices/cc12m-val.tsv" \
    --save-frequency 1 \
    --save-most-recent \
    --report-to tensorboard \
    --warmup 2000 \
    --batch-size=128 \
    --accum-freq 2 \
    --epochs=32 \
    --workers=6 \
    --model RN50 \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 100 \
    --name "clip-cc12m/RN50-cc12m-lso-real-only-s0"
```
We used an effective batch size for our CC3M / CC12M models of 2048 (i.e., 128 samples per GPU across 8 GPUs and
gradient accumulation frequency of 2).

### Supervised Classifier

For training supervised classifiers, you can use this SLURM [script](slurm/train-supervised.sh). Alternatively, you can
run the commands manually, e.g.:
```bash
python scripts/train_combined_captions.py "rn50-clip-lso-real-only-0" \
    --model rn50-clip \
    --seed 0 \
    --train_index_path "data/indices/combined-captions-train-real-only.tsv" \
    --val_index_path "data/indices/combined-captions-val-real-only.tsv" \
    --in_class_index_path "data/imagenet_class_index.json" \
    --class_mapping_path "data/in_to_dn_mapping.json" \
    --num_workers 24 \
    --ws_path supervised \
    --learning_rate 0.01 \
    --max_epochs 90
```

## Evaluation

### Classification

To evaluate the classification performance of our CLIP models, you can either use this SLURM [script](slurm/evaluate-clip.sh)
or run manually using:
```bash
python scripts/evaluate_domainnet_lso_openai.py \
    --model RN50 \
    --domain clipart \
    --out_path deps/open_clip/logs/clip/RN50-lso-real-only-0/eval \
    --domainnet_path data/domainnet \
    --imagenet_path <path/to/imagenet> \
    --num_workers 6 \
    --ckpt_files $(for e in {0..32}; do echo "deps/open_clip/logs/clip/RN50-lso-real-only-0/checkpoints/epoch_$e.pt"; done)
```
The evaluation results will be stored as a JSON file in the directory specified by `out_path`.

To evaluate the performance of our supervised classifiers, you can either use this SLURM [script](slurm/evaluate-supervised.sh)
or run the following:
```bash
python scripts/evaluate_domainnet_supervised_lso.py \
    --model rn50-clip \
    --domain clipart \
    --out_path "supervised/checkpoints/rn50-clip-lso-real-only-0/eval" \
    --domainnet_path data/domainnet \
    --num_workers 6 \
    --ckpt_files \
        supervised/checkpoints/rn50-clip-real-only-0/epoch=0-step=0.ckpt \
        supervised/checkpoints/rn50-clip-real-only-0/epoch=4-step=$STEP.ckpt \
        ...
```
For details about the `STEP` variable, please refer to the [script](slurm/evaluate-supervised.sh).

### Feature Sharing

To evaluate feature sharing, we first need to train the SAEs. Note that we conducted our SAE experiments on our CC12M
models since the SAEs extracted poor features for the ImageNet-Captions models. You can run the SAE training either via
the SLURM [script](slurm/train-sae.sh) or use:
```bash
python scripts/train_sae.py \
    --img_enc_name RN50 \
    --out_dir "deps/open_clip/logs/clip-cc12m/RN50-lso-real-only-s0/sae" \
    --domainnet_path data/domainnet \
    --cc12m_path <path/to/cc12m> \
    --ckpt_path "deps/open_clip/logs/clip-cc12m/RN50-lso-real-only-s0/checkpoints/epoch_32.pt" \
    --num_workers 6 \
    --train_sae_bs 2048 \
    --ckpt_freq 100000000 \
    --val_freq 5000000 \
    --l1_coeff 1e-4
```

Afterwards, you can evaluate the amount of feature sharing for a given model using:
```bash
python scripts/analyze_sae_features.py \
    --domain clipart \
    --domainnet_path data/domainnet \
    --model_path "deps/open_clip/logs/clip-cc12m/RN50-lso-real-only-s0" \
    --num_workers 6
```

### Separation of Quickdraw Embeddings

To reproduce UMAP plots (Figure 5), you can run the following for the standard model (Figure 5a):
```bash
python scripts/embedding_analysis.py \
    --model RN50 \
    --ckpt_files deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-s0/checkpoints/epoch_32.pt \
    --out_path deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-s0/embedding_analysis \
    --domainnet_path data/domainnet \
    --umap
``` 
and this for the aligned model (Figure 5b):
```bash
python scripts/embedding_analysis.py \
    --model RN50 \
    --model_dir deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-aligned-s0/checkpoints/epoch_32.pt \
    --out_path deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-s0/embedding_analysis \
    --domainnet_path data/domainnet \
    --umap
``` 

### Representational Similarity

To compute the CKA-based representational similarity (Figure 6a), you can run:
```bash
python scripts/representational_analysis.py \
    --model RN50 \
    --model_dir deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-aligned-s0 \
    --domainnet_path data/domainnet \
``` 

### Circuit Similarity

To compute circuits on the aligned quickdraw model (Section 6), you can run:
```bash
python scripts/compute_circuits.py \
    --model RN50 \
    --model_dir deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-aligned-s0 \
    --domainnet_path data/domainnet
```

After computing the circuits, you can evaluate the node similarity (Figure 6b) using:
```bash
python scripts/compute_node_similarity.py \
    --model_dir deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-aligned-s0
```
and circuit similarity (Figure 6c) using:
```bash
python scripts/compute_circuit_similarity.py \
    --model_dir deps/open_clip/logs/clip/RN50-lso-cipqrs-noquickdrawclasses-aligned-s0 \
    --plot
```
