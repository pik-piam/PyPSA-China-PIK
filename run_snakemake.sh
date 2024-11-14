#! /bin/bash    
#SBATCH --export=PATH,LD_LIBRARY_PATH,GUROBI_HOME
#SBATCH --job-name=Snakemake
#SBATCH --output=logs_slurm/submit/test-%j.out
#SBATCH --error=logs_slurm/submit/test-%j.er
#SBATCH --mem=2048
#SBATCH --qos=priority
#SBATCH --cpus-per-task=8 

#set up the tunnel to the login nodes (does not work well as a subprocess, even with source)
source activate pypsa-china
PORT=1080
ssh -fN -D $PORT $USER@login01 &

export https_proxy=socks5://127.0.0.1:$PORT
export SSL_CERT_FILE=/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08
export GRB_CAFILE=/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08

# Add gurobi references
export GUROBI_HOME="/p/projects/rd3mod/gurobi1103/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE=/p/projects/rd3mod/gurobi_rc/gurobi.lic
export GRB_CURLVERBOSE=1

echo "launching snakemake"
# launch workflow with the PIK hpc_profile (this generates the relevant sbatch commands)
snakemake --profile config/pik_hpc_profile/