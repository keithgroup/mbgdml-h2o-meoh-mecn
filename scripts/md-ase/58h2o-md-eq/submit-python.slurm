#!/bin/bash
#SBATCH --job-name=58h2o-tau_0.01-1
#SBATCH --output=58h2o-mbgdml-nvt-init_100-tau_0.01-1.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=1-00:00:00
#SBATCH --cluster=smp
#SBATCH --partition=smp

# Initialize conda environment
module purge
export PATH=/ihome/jkeith/amm503/miniconda3/condabin:$PATH
source activate /ihome/jkeith/amm503/miniconda3/envs/mbgdml-torch

# Ray setup
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")  # e.g., smp-n28
nodes_array=($nodes)  # e.g., smp-n28
node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" ray start --head --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block & sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done


cd ${SLURM_SUBMIT_DIR}
python -u nvt-beren-from-traj-initv.py

crc-job-stats.py
