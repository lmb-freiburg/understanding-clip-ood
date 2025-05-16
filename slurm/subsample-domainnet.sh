#!/bin/bash

# Define the partitition on which the job shall run.
#SBATCH --partition <TODO>  # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name xclip          # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output <TODO>/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error <TODO>/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 32GB
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:1

# Get mail notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<TODO>

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
# TODO: activate your environment

# Running the job
start=`date +%s`

# Natural-only
python scripts/subsample_domainnet_lso.py --real_only --indices_path data/indices --domainnet_path data/domainnet

# Quickdraw CG high-diversity with aligned captions (Section 7)
python scripts/subsample_domainnet_lso.py --exclude quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet/aligned-captions --aligned_captions

for DOMAIN in clipart infograph painting quickdraw sketch; do
    # CG low-diversity
    python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --subsample --indices_path data/indices --domainnet_path data/domainnet
    # CG high-diversity
    python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --subsample --indices_path data/indices --domainnet_path data/domainnet
    # Leave-out-domain
    python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --exclude_domains $DOMAIN --subsample --indices_path data/indices --domainnet_path data/domainnet
    # CG low-diversity (with test classes)
    python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --subsample --pseudo_exclude --indices_path data/indices --domainnet_path data/domainnet
    # CG high-diversity (with test classes)
    python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --subsample --pseudo_exclude --indices_path data/indices --domainnet_path data/domainnet

    # upper bound interpolation experiments (Figure 13), comment in if needed
    # # CG low-diversity (upper bound interpolations)
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.05 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.1 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.2 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.4 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.6 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --single_domain --allow 0.8 --subsample --indices_path data/indices --domainnet_path data/domainnet

    # # CG high-diversity (upper bound interpolations)
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.05 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.1 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.2 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.4 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.6 --subsample --indices_path data/indices --domainnet_path data/domainnet
    # python scripts/subsample_domainnet_lso.py --exclude $DOMAIN --allow 0.8 --subsample --indices_path data/indices --domainnet_path data/domainnet
done

# domain interpolation experiments (Figure 3 & 8), comment in if needed
# # sketch interpolation experiments
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains clipart infograph painting --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains clipart infograph quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains clipart painting quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains infograph painting quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains infograph painting --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains infograph quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains painting quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains infograph --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude sketch --exclude_domains quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet

# # clipart interpolation experiments
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph painting quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph painting sketch --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph quickdraw sketch --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains painting quickdraw sketch --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph painting --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains painting quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains infograph --subsample --indices_path data/indices --domainnet_path data/domainnet
# python scripts/subsample_domainnet_lso.py --exclude clipart --exclude_domains quickdraw --subsample --indices_path data/indices --domainnet_path data/domainnet

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime second

