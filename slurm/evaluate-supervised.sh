#!/bin/bash

# Define the partitition on which the job shall run.
#SBATCH --partition <TODO>  # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name xclip-eval        # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output <TODO>/logs/%x-%A-%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error <TODO>/logs/%x-%A-%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 8 # number of cores
#SBATCH --gres=gpu:1

# Get mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<TODO>
#SBATCH -a 1-45

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
# TODO: activate your environment

# Running the job
start=`date +%s`

EXPDIR=supervised

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
    STEPS=11300
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=cr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ipqrs
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 2 ]; then
    DOMAIN=infograph
    STEPS=11335
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=ir
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=cpqrs
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 3 ]; then
    DOMAIN=painting
    STEPS=11605
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=pr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ciqrs  
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 4 ]; then
    DOMAIN=quickdraw
    STEPS=12925
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=qr
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=ciprs  
    fi
elif [ $(($SLURM_ARRAY_TASK_ID % 5)) -eq 0 ]; then
    DOMAIN=sketch
    STEPS=11575
    if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
        DOMAINS=rs
    elif [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
        DOMAINS=cipqrs
    elif [ $SLURM_ARRAY_TASK_ID -le 15 ]; then
        DOMAINS=cipqr
    fi
fi

MODEL=rn50-clip
SUFFIX=classes
# SUFFIX=classes-pseudo
IDENTIFIER="lso-$DOMAINS-no$DOMAIN$SUFFIX-$SEED"
# IDENTIFIER="real-only-$SEED"  # uncomment for natural-only eval
# STEPS=10670  # uncomment for natural-only eval


CKPT=$(for e in {1..18}; do echo "$EXPDIR/checkpoints/$MODEL-$IDENTIFIER/epoch=$((5 * $e - 1))-step=$(($STEPS * $e)).ckpt"; done)
OUTPATH="$EXPDIR/checkpoints/$MODEL-$IDENTIFIER/eval"
# OUTPATH="$EXPDIR/checkpoints/$MODEL-$IDENTIFIER/eval-$DOMAIN"  # uncomment for natural-only eval

if [ ! -d $OUTPATH ]; then
    mkdir $OUTPATH
fi

CKPT="$EXPDIR/checkpoints/$MODEL-$IDENTIFIER/epoch=0-step=0.ckpt
$CKPT"

python scripts/evaluate_domainnet_supervised_lso.py --model $MODEL --domain $DOMAIN --out_path "$OUTPATH" \
    --domainnet_path data/domainnet --num_workers 6 --ckpt_files $CKPT

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime second
