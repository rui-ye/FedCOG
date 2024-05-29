import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from config import get_args
from model import get_model
from utils import setup_seed, compute_acc
from prepare_data import get_dataloader



def local_train_scaffold(args, train_loaders, nets_this_round, auxilary_model_list, global_model, auxilary_global_model):
    

    # scaffold delta
    total_delta = copy.deepcopy(global_model.state_dict())#Δx
    for key in total_delta:
        total_delta[key] = 0.0
    
    auxilary_global_model.cuda()
    global_model.cuda()
    
    # Conduct local model training
    for client_id, model in nets_this_round.items():
        model.cuda()
        auxilary_model = auxilary_model_list[client_id].cuda()# ci

        auxilary_global_para = auxilary_global_model.state_dict()# x paramter
        auxilary_model_para = auxilary_model.state_dict()# yi parameter

        train_loader = train_loaders[client_id]
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        model.train()

        cnt = 0
        iterator = iter(train_loader)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, target = next(iterator)
            cnt += 1
            x, target = x.cuda(), target.long().cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
    
            net_para = model.state_dict()
            for key in net_para:
                net_para[key] = net_para[key] - args.lr * (auxilary_global_para[key] - auxilary_model_para[key])#yi ←yi −ηl (gi(yi) −ci + c)#少一个c
            model.load_state_dict(net_para)
        
        auxilary_new_para = auxilary_model.state_dict()#ci+
        auxilary_delta_para = copy.deepcopy(auxilary_model.state_dict())
        global_model_para = global_model.state_dict()#x
        net_para = model.state_dict()
        #auxilary_new_para: ci+, auxilary_model_para: ci
        for key in net_para:
            auxilary_new_para[key] = auxilary_new_para[key] - auxilary_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)#ci+
            auxilary_delta_para[key] = auxilary_new_para[key] - auxilary_model_para[key] #ci+ −ci
        auxilary_model.load_state_dict(auxilary_new_para)# ci <- ci+
        auxilary_model.to('cpu')

        for key in total_delta:
            total_delta[key] += auxilary_delta_para[key] #Δx = sumΔx^i

        model.to('cpu')
    global_model.to('cpu')

    for key in total_delta:
        total_delta[key] /= args.n_parties# len(nets_this_round)#Δx = Δx/N
    auxilary_global_para = auxilary_global_model.state_dict()
    for key in auxilary_global_para:
        if auxilary_global_para[key].type() == 'torch.LongTensor':
            auxilary_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif auxilary_global_para[key].type() == 'torch.cuda.LongTensor':
            auxilary_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            auxilary_global_para[key] += total_delta[key]
    auxilary_global_model.load_state_dict(auxilary_global_para)
    

args, cfg = get_args()
print(args)
setup_seed(args.init_seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

train_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

model = get_model(args)
        
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()
local_models = []

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    
auxilary_global_model = copy.deepcopy(global_model) 
auxilary_model_list = [copy.deepcopy(global_model) for _ in range(args.n_parties)]

best_acc = 0
for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    global_w = global_model.state_dict()        # Global Model Initialization

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.load_state_dict(global_w)
    
    # Local Model Training
    local_train_scaffold(args, train_local_dls, nets_this_round, auxilary_model_list, global_model, auxilary_global_model)
    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round==0 or args.sample_fraction<1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')
    

    # Model Aggregation
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.load_state_dict(global_w)          # Update the global model
    

    acc = compute_acc(global_model, test_dl)
    if best_acc < acc:
        best_acc = acc
    print('>> Round {} | Current Acc: {:.5f}, Best Acc: {:.5f}'.format(round, acc, best_acc))
    print('-'*80)

    if (round+1)%args.comm_round == 0 and args.save_model:
        torch.save(global_w, f"./models/saved_model/{args.dataset}/scaffold_{args.dataset}_{args.model}_{args.partition}{args.beta}_c{args.n_parties}_it{args.num_local_iterations}_p{args.sample_fraction}_{round+1}_{acc:.5f}.pkl")

 