import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model', type=str, default='resnet18_gn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--personalized_learning_rate', type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--n_domain_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=55, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default=2, help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--lambda_1', type=float, default=0.01, help='fedprox hyperparameter')

    # For FEMNIST dataset
    parser.add_argument('--femnist_sample_top', type=int, default=1, help='whether to sample top clients from femnist')
    parser.add_argument('--femnist_train_num', type=int, default=20, help='how many clients from femnist are sampled')
    parser.add_argument('--femnist_test_num', type=int, default=20, help='number of testing clients from femnist')

    # Important paranmeters for generating
    parser.add_argument('--adv', default=0.1, type=float, help='scaling factor for adv loss')
    parser.add_argument('--bn', default=0.0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=1.0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--generator', action="store_true", help="whether to use generater.")
    parser.add_argument('--gen_aug', action="store_true", help="whether to use augumentation during image synthesizing.")
    parser.add_argument('--gen_downsample', action="store_true", help="whether to save the global model.")
    parser.add_argument('--lr_g', default=1e-2, type=float, help='initial learning rate for generation')
    parser.add_argument('--js_T', default=1, type=float, help='temperature for js div')
    parser.add_argument('--g_steps', default=500, type=int, metavar='N', help='number of iterations for generation')
    parser.add_argument('--nz', default=256, type=int, metavar='N', help='dimension of noise')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    parser.add_argument('--T', default=1, type=float, help='temperature for kd training')
    parser.add_argument('--start_round', default=50, type=int, help='the first round to implement generation for fedcog')
    parser.add_argument('--double_gen', action="store_true", help='generate twice each round')
    parser.add_argument('--load_path', type=str, default=None, help='load model checkpoint')
    parser.add_argument('--save_model', action="store_true", help="whether to save the global model.")
    parser.add_argument('--gen_complement', action="store_true", help="whether to generate according to distribution")

    # FedAvgM
    parser.add_argument('--server_momentum', type=float, default=0.0)

    # FedProx / MOON
    parser.add_argument('--mu', type=float, default=0.01)

    # FedDecorr
    parser.add_argument('--decorr_beta', type=float, default=0.1, help='parameter for loss term in Feddecor')

    # FedExP
    parser.add_argument('--exp_eps', type=float, default=1e-3, help='parameter for FedEXP in model aggregation')

    # FedDyn
    parser.add_argument('--dyn_alpha', type=float, default=0.01, help='parameter for FedDyn')

    # FedADAM
    parser.add_argument('--adam_server_momentum_1', type=float, default=0.9, help='first order parameter for fedadam.')
    parser.add_argument('--adam_server_momentum_2', type=float, default=0.99, help='second order parameter for fedadam.')
    parser.add_argument('--adam_server_lr', type=float, default=1.0, help='server learning rate for fedadam.')
    parser.add_argument('--adam_tau', type=float, default=0.001, help='tau for fedadam.')

    # FedSAM
    parser.add_argument('--sam_rho', type=float, default=0.05, help='rho for fedsam.')

    # VHL
    parser.add_argument('--VHL_alpha', default=1.0, type=float)
    parser.add_argument('--VHL_feat_align', action="store_true", help='if aligning feature in training')
    parser.add_argument('--VHL_generative_dataset_root_path', default='/GPFS/data/yaxindu/FedHomo/VHL/data_preprocessing/generative/dataset/', type=str)
    parser.add_argument('--VHL_dataset_batch_size', default=128, type=int)
    parser.add_argument('--VHL_dataset_list', default="Gaussian_Noise", type=str, help="either Gaussian_Noise or style_GAN_init")
    parser.add_argument('--VHL_align_local_epoch', default=5, type=int)
    
    # FedReg
    parser.add_argument('--reg_gamma', default=0.5, type=float)
    parser.add_argument('--reg_iter', default=10, type=int)
    parser.add_argument('--reg_eta', default=1e-3, type=float)
    
    # DisTrans
    parser.add_argument('--gen_lr', default=1e-2, type=float)
    parser.add_argument('--distrans_alpha', default=0.3, type=float)
    parser.add_argument('--nn_agg', default=True, action="store_false")
    parser.add_argument('--offset_eval', default=False, action="store_true")

    args = parser.parse_args()
    cfg = dict()
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'yahoo_answers'}:
        cfg['classes_size'] = 10
        cfg["channel"] = 3
        cfg["image_size"] = 32
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
        cfg["channel"] = 3
        cfg["image_size"] = 32
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
        cfg["channel"] = 3
        cfg["image_size"] = 64
    elif args.dataset == 'pacs':
        cfg['classes_size'] = 7
        cfg["channel"] = 3
        cfg["image_size"] = 64
    elif args.dataset == 'femnist':
        cfg['classes_size'] = 62
        cfg["channel"] = 1
        cfg["image_size"] = 28
    elif args.dataset == 'flair':
        cfg['classes_size'] = 17
        cfg["channel"] = 3
    elif args.dataset ==  'fashionmnist':
        cfg['classes_size'] = 10
        cfg["channel"] = 1
        cfg["image_size"] = 28
    else:
        args.image_size = cfg["image_size"]
    args.channel = cfg["channel"]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg
