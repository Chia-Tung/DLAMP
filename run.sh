#!/usr/bin/env bash

#PJM -L rscunit=rscunit_pg01            # Resource unit name
#PJM -L rscgrp=gpu-rd-large             # Resource group name
#PJM -L node=1                          # GPU node
#PJM -L vnode=1                         # number of Nodes
#PJM -L vnode-core=32                   # CPU cores
#PJM --mpi proc=32                      # MPI processes
#PJM -L gpu=8                           # GPU Cards
#PJM -L elapse=720:00:00                # elapse time
#PJM -o 'sys_log/test.%j.out'           # write log out to log directory
#PJM -e 'sys_log/test.%j.err'           # write error message to log directory

# activate virtual env
. /nwpr/wfc/com134/.bashrc

module use /package/x86_64/nvidia/hpc_sdk/23.1/modulefiles
module load nvhpc/23.1
module list

CUDA_PATH=/package/x86_64/nvidia/hpc_sdk/23.1/Linux_x86_64/23.1/cuda/11.8
export PATH=${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}

# Print CUDA library path
echo "CUDA Library Path:"
echo $LD_LIBRARY_PATH
echo -e "\nGPU Server Information:"
hostname
echo -e "\nGPU Status:"
nvidia-smi
echo -e "\ntest python modules"
python -V
python -c "import torch; print(f'pytorch {torch.__version__}'); import lightning; print(f'lightning {lightning.__version__}')"

# job run
python train.py