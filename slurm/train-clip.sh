#!/bin/bash

# Define the partitition on which the job shall run.
#SBATCH --partition <TODO>  # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name xclip        # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output <TODO>/logs/%x-%A-%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error <TODO>/logs/%x-%A-%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 128GB
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH --cpus-per-task 8 # number of cores
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -a 1-45  # run all 15 experiments, three seeds each

# Get mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<TODO>

cd deps/open_clip  # code is designed to be run directly from OpenCLIP directory
echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
# TODO: activate your environment

# Running the job
start=`date +%s`

EXPDIR=clip

# export MASTER_PORT=12802
export MASTER_PORT=$((12802 + $SLURM_ARRAY_TASK_ID))

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


master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export PYTHONPATH="$PYTHONPATH:$PWD/src"

MODEL=RN50
# MODEL=ViT-S-32
# MODEL=swin_tiny_patch4_window7_224

# RN50 uses more VRAM => lower batch size and gradient accumulation required
# batch size is per gpu and needs to be adjusted when changing the number of GPUs
if [ $MODEL == "RN50" ]; then
    BATCH_SIZE=128
    ACC_FREQ=2
else
    BATCH_SIZE=256
    ACC_FREQ=1
fi


# NOTE: to also train natural-only or upper bound settings you need to manipulate the suffix and data id
# for upper bound training, uncomment the suffix line below and adjust the job array from 1-45 to 1-10,16-25,31-40
# as there are no leave-out-domain upper bounds
# for natural-only training uncomment the data id line below and adjust the job array from 1-45 to 1,16,31
SUFFIX=classes
# SUFFIX=classes-pseudo  # uncomment for upper bound training
DATAID="lso-$DOMAINS-no$DOMAIN$SUFFIX"
# DATAID="lso-real-only"  # uncomment for natural-only training

IDENTIFIER="$DATAID-s$SEED"
echo "Identifier: $IDENTIFIER"

if [ -d "logs/$EXPDIR/$MODEL-$IDENTIFIER" ]; then
    if [ -f "logs/$EXPDIR/$MODEL-$IDENTIFIER/checkpoints/epoch_32.pt" ]; then
        echo "Experiment logs/$EXPDIR/$MODEL-$IDENTIFIER exists and is fully trained. Aborting."
        exit 0
    elif [ -f "logs/$EXPDIR/.RUNNING_$MODEL-$IDENTIFIER" ]; then
        echo "Experiment logs/$EXPDIR/$MODEL-$IDENTIFIER exists and is running. Aborting."
        exit 0
    else
        if [ -f "logs/$EXPDIR/$MODEL-$IDENTIFIER/checkpoints/epoch_latest.pt" ]; then
            echo "Experiment logs/$EXPDIR/$MODEL-$IDENTIFIER exists, is not fully trained and is not running. Continuing."
            RESUME="--resume latest"
        else
            echo "Experiment logs/$EXPDIR/$MODEL-$IDENTIFIER exists, is not fully trained (< 1 epochs) and is not running. Removing."
            rm -r "logs/$EXPDIR/$MODEL-$IDENTIFIER"
            RESUME=
        fi
    fi
fi

touch "$EXPDIR/.RUNNING_$MODEL-$IDENTIFIER"
srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --train-data "../../data/indices/combined-captions-train-$DATAID.tsv" \
    --val-data "../../data/indices/combined-captions-val-$DATAID.tsv" \
    --save-frequency 1 \
    --save-most-recent \
    --report-to tensorboard \
    --lr 0.001 \
    --warmup 500 \
    --batch-size=$BATCH_SIZE \
    --accum-freq $ACC_FREQ \
    --epochs=32 \
    --workers=6 \
    --model "$MODEL" \
    --seed $SEED \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 50 \
    --name "$EXPDIR/$MODEL-$IDENTIFIER" \
    $RESUME
rm "$EXPDIR/.RUNNING_$MODEL-$IDENTIFIER"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime second
