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


def local_train_moon(args, nets_this_round, nets_previous, global_net, train_local_dls):
    
    global_net.eval()
    global_net.cuda()

    for net_id, net in nets_this_round.items():
        prev_net = nets_previous[net_id]
        prev_net.eval()
        train_local_dl = train_local_dls[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)

        criterion = torch.nn.CrossEntropyLoss().cuda()
        ce_loss = torch.nn.CrossEntropyLoss().cuda()
        cos = torch.nn.CosineSimilarity(dim=-1)
        net.cuda()
        prev_net.cuda()
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            pro1 = net.base(x).squeeze()
            out = net.classifier(pro1)
            with torch.no_grad():
                pro2 = global_net.base(x).squeeze()
                pro3 = prev_net.base(x).squeeze()

            loss1 = criterion(out, target)

            pos = cos(pro1, pro2)
            logits = pos.reshape(-1,1)
            nega = cos(pro1, pro3)
            logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
            logits /= 0.5
            labels = torch.zeros(x.size(0)).cuda().long()
            loss2 = args.mu * ce_loss(logits, labels)

            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
        net.to('cpu')
        prev_net.to('cpu')
    global_net.to('cpu')

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

train_local_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)

model = get_model(args)
        
global_model = model(cfg['classes_size'])
local_models = []

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))

if args.server_momentum:
    moment_v = copy.deepcopy(global_model.state_dict())
    for key in moment_v:
        moment_v[key] = 0
    
best_acc = 0
for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')

    global_w = global_model.state_dict()        # Global Model Initialization

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_previous = copy.deepcopy(nets_this_round)
    for net in nets_this_round.values():
        net.load_state_dict(global_w)
    
    # Local Model Training
    local_train_moon(args, nets_this_round, nets_previous, global_model, train_local_dls)
    # Aggregation Weight Calculation
    total_data_points = sum([client_num_samples[r] for r in party_list_this_round])
    fed_avg_freqs = [client_num_samples[r] / total_data_points for r in party_list_this_round]
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
        torch.save(global_w, f"./models/saved_model/{args.dataset}/moon_{args.dataset}_{args.model}_{args.partition}{args.beta}_c{args.n_parties}_it{args.num_local_iterations}_p{args.sample_fraction}_{round+1}_{acc:.5f}.pkl")
