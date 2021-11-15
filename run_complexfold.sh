# Description: AlphaFold non-docker version
# Author: Matthias Uthoff




# Only change the section Defaults !





usage() {
        echo ""
        echo "Please make sure all required parameters are given"
        echo "Usage: $0 <OPTIONS>"
        echo "Required Parameters:"
        echo "-f <fasta_path>   Path to a FASTA file containing one sequence"
        echo "Optional Parameters:"
        echo "-d <data_dir>     Path to directory of supporting data"
        echo "-o <output_dir>   Path to a directory that will store the results."
        echo "-m <model_names>  Names of models to use (a comma separated list)"
        echo "-y <thoroughness>  Thoroughness of prediction. Sets <preset>, <num_recycle>, <recycling_tolerance> and <num_seeds> to certain values. You can still overwrite them. Can either be: low, alphafold, medium, high, extreme."
        echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo "-n <openmm_threads>   OpenMM threads (default: all available cores)"
        echo "-g <use_gpu>      Enable NVIDIA runtime to run with GPUs (default: True)"
        echo "-a <gpu_devices>  Comma separated list of devices to pass to 'CUDA_VISIBLE_DEVICES' (default: 0)"
        echo "-p <preset>       Choose preset model configuration - no ensembling and smaller genetic database config (reduced_dbs), no ensembling and full genetic database config  (full_dbs) or full genetic database config and 8 model ensemblings (casp14)"
        echo "-u <amber_accel> Hardware used to carry out amber: Either CPU or CUDA. (default: CUDA)."
        echo "-r <num_recycle> Number of recycling during prediction."
        echo "-l <recycling_tolerance> Tolerance for deciding when to stop recycling (Ca-RMS)."
        echo "-s <random_seeds> The random seed for the data pipeline, comma-separated list. By default, this is randomly generated. Note that even if this is set, Alphafold may still not be deterministic, because processes like GPU inference are nondeterministic."
        echo "-x <num_seeds> Number of random seeds to use."
        echo "-i <focus_region> Focus on position x through y while deciding which result to keep. Uses the mean pLDDT of that region. For complexes, concatenate sequences and count from the very beginning."
        echo "-b <benchmark>    Run multiple JAX model evaluations to obtain a timing that excludes the compilation time, which should be more indicative of the time required for inferencing many proteins (default: 'False')"
        echo "-c <complex_mode> Use ptm models (better for complexes) instead of usual ones."
        echo "-w <write_features_models> Write features.pkl and model resut pkls."
        echo ""
        exit 1
}

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################
conda activate af2







# Defaults
## Path of ComplexFold script
alphafold_script=/home/matthias/ComplexFold/run_complexfold.py

## Dir to the databases
data_dir=/home/matthias/HDD/Alphafold_DBs

## Path of each database
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
small_bfd_database_path="$data_dir/small_bfd/bfd-first_non_consensus_sequences.fasta"
mgnify_database_path="$data_dir/mgnify/mgy_clusters.fa"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"
pdb70_database_path="$data_dir/pdb70/pdb70"
uniclust30_database_path="$data_dir/uniclust30/UniRef30_2021_06"
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
## Library for pre-computed MSAs - Please make this dir before use!
msa_library_dir="$data_dir/msa_library"

## Output dir
output_dir=/home/matthias/Documents/Structures/Alphafold

## Which GPU device to use. Usually you can keep that as 0. If you do not have a GPU set use_gpu=false
gpu_devices=0
use_gpu=true

## Relaxation is accelerated by the GPU (set to "CUDA") or runs on the the CPU (set to "CPU").
amber_accel="CUDA"

# Further variables defining what a default run of ComplexFold does. You should leav them.
max_template_date=$( date +%Y-%m-%d )
random_seeds=-1
focus_region=""
benchmark=false
thoroughness=alphafold
complex_mode=false
write_features_models=false

# Binary path (change me if required)
hhblits_binary_path=$(which hhblits)
hhsearch_binary_path=$(which hhsearch)
jackhmmer_binary_path=$(which jackhmmer)
kalign_binary_path=$(which kalign)

# No reason to change anything below this
################################################################################################







while getopts "d:o:m:y:f:t:g:n:a:p:u:r:l:s:x:i:bcw" opt
do
    case $opt in
        d)
            data_dir=$OPTARG
            ;;
        o)
            output_dir=$OPTARG
            ;;
        m)
            model_names=$OPTARG
            ;;
        y)
            thoroughness=$OPTARG
            ;;
        f)
            fasta_path=$OPTARG
            ;;
        t)
            max_template_date=$OPTARG
            ;;
        g)
            use_gpu=$OPTARG
            ;;
        n)
            openmm_threads=$OPTARG
            ;;
        a)
            gpu_devices=$OPTARG
            ;;
        p)
            preset=$OPTARG
            ;;
        u)
            amber_accel=$OPTARG
            ;;
        r)
            num_recycle=$OPTARG
            ;;
        l)
            recycling_tolerance=$OPTARG
            ;;
        s)
            random_seeds=$OPTARG
            ;;
        x)
            num_seeds=$OPTARG
            ;;
        i)
            focus_region=$OPTARG
            ;;
        b)
            benchmark=true
            ;;
        c)
            complex_mode=true
            ;;
        w)
            write_features_models=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

# Parse input
# Via thorougness
if [[ "$thoroughness" == "low" ]] ; then
    preset_="reduced_dbs"
    num_recycle_=2
    recycling_tolerance_=0.5
    num_seeds_=1
elif [[ "$thoroughness" == "alphafold" ]] ; then
    preset_="full_dbs"
    num_recycle_=3
    recycling_tolerance_=0.0
    num_seeds_=1
elif [[ "$thoroughness" == "medium" ]] ; then
    preset_="full_dbs"
    num_recycle_=10
    recycling_tolerance_=0.5
    num_seeds_=5
elif [[ "$thoroughness" == "high" ]] ; then
    preset_="full_dbs"
    num_recycle_=15
    recycling_tolerance_=0.33
    num_seeds_=20
elif [[ "$thoroughness" == "extreme" ]] ; then
    preset_="full_dbs"
    num_recycle_=20
    recycling_tolerance_=0.33
    num_seeds_=30
else
    usage
fi
if [[ "$preset" == "" ]] ; then
    preset=$preset_
fi
if [[ "$num_recycle" == "" ]] ; then
    num_recycle=$num_recycle_
fi
if [[ "$recycling_tolerance" == "" ]] ; then
    recycling_tolerance=$recycling_tolerance_
fi
if [[ "$num_seeds" == "" ]] ; then
    num_seeds=$num_seeds_
fi

# Other inputs
if [[ "$model_names" == "" ]] ; then
    if [[ "$complex_mode" == true ]] ; then
        model_names="model_1_ptm,model_2_ptm,model_3_ptm,model_4_ptm,model_5_ptm"
    else
        model_names="model_1,model_2,model_3,model_4,model_5"
    fi
fi
if [[ "$fasta_path" == "" ]] ; then
    usage
fi
if [[ "$preset" != "full_dbs" && "$preset" != "casp14" && "$preset" != "reduced_dbs" ]] ; then
    echo "Unknown preset! Using default ('full_dbs')"
    preset="full_dbs"
fi

# Export ENVIRONMENT variables and set CUDA devices for use
# CUDA GPU control
export CUDA_VISIBLE_DEVICES=-1
if [[ "$use_gpu" == true ]] ; then
    export CUDA_VISIBLE_DEVICES=0

    if [[ "$gpu_devices" ]] ; then
        export CUDA_VISIBLE_DEVICES=$gpu_devices
    fi
fi

# OpenMM threads control
if [[ "$openmm_threads" ]] ; then
    export OPENMM_CPU_THREADS=$openmm_threads
fi

# TensorFlow control
export TF_FORCE_UNIFIED_MEMORY='1'

# JAX control
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

# Run AlphaFold with required parameters
# 'reduced_dbs' preset does not use bfd and uniclust30 databases
if [[ "$preset" == "reduced_dbs" ]]; then
    $(python $alphafold_script \
    --hhblits_binary_path=$hhblits_binary_path \
    --hhsearch_binary_path=$hhsearch_binary_path \
    --jackhmmer_binary_path=$jackhmmer_binary_path \
    --kalign_binary_path=$kalign_binary_path \
    --small_bfd_database_path=$small_bfd_database_path \
    --mgnify_database_path=$mgnify_database_path \
    --template_mmcif_dir=$template_mmcif_dir \
    --obsolete_pdbs_path=$obsolete_pdbs_path \
    --pdb70_database_path=$pdb70_database_path \
    --uniref90_database_path=$uniref90_database_path \
    --data_dir=$data_dir \
    --msa_library_dir=$msa_library_dir \
    --output_dir=$output_dir \
    --fasta_paths=$fasta_path \
    --model_names=$model_names \
    --max_template_date=$max_template_date \
    --preset=$preset \
    --benchmark=$benchmark \
    --write_features_models=$write_features_models \
    --amber_accel=$amber_accel \
    --recycling_tolerance=$recycling_tolerance \
    --num_recycle=$num_recycle \
    --random_seeds=$random_seeds \
    --num_seeds=$num_seeds \
    --focus_region=$focus_region \
    --logtostderr)
else
    $(python $alphafold_script \
    --hhblits_binary_path=$hhblits_binary_path \
    --hhsearch_binary_path=$hhsearch_binary_path \
    --jackhmmer_binary_path=$jackhmmer_binary_path \
    --kalign_binary_path=$kalign_binary_path \
    --bfd_database_path=$bfd_database_path \
    --mgnify_database_path=$mgnify_database_path \
    --template_mmcif_dir=$template_mmcif_dir \
    --obsolete_pdbs_path=$obsolete_pdbs_path \
    --pdb70_database_path=$pdb70_database_path \
    --uniclust30_database_path=$uniclust30_database_path \
    --uniref90_database_path=$uniref90_database_path \
    --data_dir=$data_dir \
    --msa_library_dir=$msa_library_dir \
    --output_dir=$output_dir \
    --fasta_paths=$fasta_path \
    --model_names=$model_names \
    --max_template_date=$max_template_date \
    --preset=$preset \
    --benchmark=$benchmark \
    --write_features_models=$write_features_models \
    --amber_accel=$amber_accel \
    --recycling_tolerance=$recycling_tolerance \
    --num_recycle=$num_recycle \
    --random_seeds=$random_seeds \
    --num_seeds=$num_seeds \
    --focus_region=$focus_region \
    --logtostderr)
fi

conda deactivate
