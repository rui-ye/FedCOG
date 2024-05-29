data_partition="noniid" # "noniid-skew-2", 'iid'
iternum=100
client=10
beta=0.1
dataset="fashionmnist"
model="simplecnn-mnist"
sample_fraction=1.0
comm_round=70
start_round=50
dir_path=./log/${dataset}_${data_partition}_${model}_beta${beta}_r${comm_round}_it${iternum}_c${client}_p${sample_fraction}
mkdir $dir_path


################ FedCOG ################
python -u fedcog.py --dataset $dataset --gpu "6" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --start_round $start_round --comm_round $comm_round --save_model > $dir_path/fedcog.log

# ################ FedAvg ################
# python -u fedavg.py --dataset $dataset --gpu "0" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --comm_round $comm_round --save_model > $dir_path/fedavg.log

# ################ FedAvgM ################
# server_momentum=0.9
# python -u fedavg.py --server_momentum $server_momentum --dataset $dataset --gpu "3" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --comm_round $comm_round --save_model > $dir_path/fedavgm_${server_momentum}.log


# ################ MOON ################
# mu=0.01
# python -u moon.py --mu $mu --dataset $dataset --gpu "0" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --comm_round $comm_round --save_model > $dir_path/moon_${mu}.log

# ################ SCAFFOLD ################
# python -u scaffold.py --dataset $dataset --gpu "0" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --comm_round $comm_round --save_model > $dir_path/scaffold.log

# ################ FedProx ################
# mu=0.01
# python -u fedprox.py --mu $mu --dataset $dataset --gpu "4" --partition $data_partition --model $model --n_parties $client --sample_fraction $sample_fraction --num_local_iterations $iternum --beta $beta --comm_round $comm_round --save_model > $dir_path/fedprox_${mu}.log


