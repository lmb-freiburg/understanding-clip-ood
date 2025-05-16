#!/bin/bash

# Define the partitition on which the job shall run.
#SBATCH --partition <TODO>  # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name xclip-sae        # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output <TODO>/logs/%x-%A-%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error <TODO>/logs/%x-%A-%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 64GB
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 8 # number of cores
#SBATCH --gres=gpu:1

# Get mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<TODO>
#SBATCH -a 6,10,11,15  # only evaluate clipart/sketch high-diversity and leave-out-domain

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
# TODO: activate your environment

# Running the job
start=`date +%s`

EXPDIR=clip-cc12m

if [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
    SEED=0
    SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID - 0))
elif [ $SLURM_ARRAY_TASK_ID -le 30 ]; then
    SEED=1
    SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID - 15))
elif [ $SLURM_ARRAY_TASK_ID -le 45 ]; then
    SEED=2
    SLURM_ARRAY_TASK_ID=$((SLURM_ARRAY_TASK_ID - 30))
fi

DOMAIN=
DOMAINS=
if [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 1 ]; then
    DOMAIN=clipart
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=cr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ipqrs
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 2 ]; then
    DOMAIN=infograph
    DOMAINS=cpqrs    
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=ir
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=cpqrs
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 3 ]; then
    DOMAIN=painting
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=pr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ciqrs  
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 4 ]; then
    DOMAIN=quickdraw
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=qr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ciprs  
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 0 ]; then
    DOMAIN=sketch
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=rs
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=cipqr
    fi
fi

MODEL=RN50
SUFFIX=classes
IDENTIFIER="lso-$DOMAINS-no$DOMAIN$SUFFIX-s$SEED"

EVALDIR=sae
OUTPATH="deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER/$EVALDIR"

if [ -d "$OUTPATH/checkpoints" ]; then
    # rm -r "$OUTPATH/checkpoints"
    rm -r "$OUTPATH/tensorboard"
    rm -r "$OUTPATH/concepts"
fi

if [ ! -d $OUTPATH ]; then
    mkdir $OUTPATH
fi

echo "Identifier: $IDENTIFIER"
if [ -d "deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER" ]; then
    if [ ! -f "deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER/checkpoints/epoch_32.pt" ]; then
        echo "Experiment deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER exists and is not fully trained. Aborting."
        exit 0
    fi
else
    echo "Experiment deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER does not exist. Aborting."
    exit 0
fi

python scripts/train_sae.py --img_enc_name $MODEL --out_dir "$OUTPATH" --domainnet_path data/domainnet \
    --ckpt_path "deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER/checkpoints/epoch_32.pt" \
    --cc12m_path <TODO>/cc12m --num_workers 6 --train_sae_bs 2048 --ckpt_freq 100000000 \
    --val_freq 5000000 --l1_coeff 1e-4 # --resample_dataset_size 100000

# optionally you can also name the discovered concepts but this is not needed for evaluation of shared features
# python scripts/name_concepts.py --img_enc_name $MODEL --out_dir "$OUTPATH" \
#     --vocab_file data/clipdissect_20k.txt \
#     --ckpt_path "deps/open_clip/logs/$EXPDIR/$MODEL-$IDENTIFIER/checkpoints/epoch_32.pt"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime second
