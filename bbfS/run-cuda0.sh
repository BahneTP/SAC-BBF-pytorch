set -ex

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XLA_FLAGS=--xla_gpu_graph_level=0

#for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
strings=("Breakout") # Qbert, Gopher, Breakout?
seed=27 
for ((j=11;j<=20;j++));
do
for game_name in "${strings[@]}";
do
    echo "iteration ${j}"
    CUDA_VISIBLE_DEVICES=0 python -m bbf.train \
        --agent=BBF \
        --gin_files=bbf/configs/BBF-100K.gin \
        --gin_bindings="DataEfficientAtariRunner.game_name=\"${game_name}\"" \
        --run_number=${j} #\
        #--agent_seed=$seed --eval_only=True --no_seeding=False
done
done
#rm -rf /tmp/online_rl/bbf/cuda0/$seed
