#!/bin/bash

# Define the partitition on which the job shall run.
#SBATCH --partition <TODO>  # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name xclip       # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output <TODO>/%x-%A-%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error <TODO>/%x-%A-%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 192GB
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 32 # number of cores
#SBATCH --gres=gpu:4

# Get mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<TODO>
#SBATCH -a 1-45  # run all 15 experiments, three seeds each

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
# TODO: activate your environment

# Running the job
start=`date +%s`

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

WORKSPACE="supervised"
SUFFIX=classes
IDENTIFIER="lso-$DOMAINS-no$DOMAIN$SUFFIX"
VAL_IDENTIFIER="lso-$DOMAINS-no$DOMAIN$SUFFIX"
MODEL=rn50-clip

if compgen -G "$WORKSPACE/checkpoints/$MODEL-$IDENTIFIER-$SEED/epoch=89-*.ckpt" > /dev/null; then
    echo "$IDENTIFIER-$SEED is fully trained. Skipping."
    exit 0
fi

if [ -d "$WORKSPACE/checkpoints/$MODEL-$IDENTIFIER-$SEED" ]; then
    echo "$IDENTIFIER-$SEED exists but is not fully trained. Removing."
    rm -r "$WORKSPACE/checkpoints/$MODEL-$IDENTIFIER-$SEED"
    rm -r "$WORKSPACE/tensorboard_logs/$MODEL-$IDENTIFIER-$SEED"
fi

python scripts/train_combined_captions.py "$MODEL-$IDENTIFIER-$SEED" --model $MODEL --seed $SEED \
	--train_index_path "data/indices/combined-captions-train-$IDENTIFIER.tsv" \
	--val_index_path "data/indices/combined-captions-val-$VAL_IDENTIFIER.tsv" \
	--in_class_index_path "data/imagenet_class_index.json" \
	--class_mapping_path "data/in_to_dn_mapping.json" \
	--num_workers 24 --ws_path $WORKSPACE  --learning_rate 0.01 --max_epochs 90 # --precision 32-true

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime second
