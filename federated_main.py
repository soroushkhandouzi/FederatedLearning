#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNN
from utils import  average_weights, exp_details
from dataset_split import get_dataset, get_user_groups
from models_fedma import pdm_prepare_weights,pdm_prepare_freq,partition_data
from models_fedma import layer_group_descent as pdm_multilayer_group_descent
from sampling import random_number_images, non_iid_unbalanced, iid_unbalanced, non_iid_balanced, iid_unbalanced



if __name__ == '__main__':
    start_time = time.time()


    # define paths
    path_project = os.path.abspath('../../Downloads')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'


    # load dataset and user groups
    train_dataset, test_dataset = get_dataset(args)
    user_groups = get_user_groups(args)
    #user_groups=get_user_groups_alpha(args)


    # BUILD MODEL
    args.model = 'cnn'
    # Convolutional neural network

    args.dataset = 'cifar'
    global_model = CNN(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    #print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for round in range(args.communication_rounds):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        for idx in idxs_users:
            print(idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger)

            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        # update global weights
        if args.comm_type == "fedavg":
            global_weights = average_weights(local_weights)
        elif args.comm_type == "fedma":
            batch_weights = pdm_prepare_weights(global_model)
            n_classes = args.net_config
            print(n_classes)
            n_classes = n_classes[-1]
            cls_freqs = partition_data(train_dataset,test_dataset,args.n_nets)
            batch_freqs = pdm_prepare_freq(cls_freqs,n_classes)
            gammas = [1.0, 1e-3, 50.0] if gamma is None else [gamma]
            sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
            sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]

            for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
                print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)
            hungarian_weights = pdm_multilayer_group_descent(
                batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it,
                gamma_layers=gamma
            )
        else:
            print("you did not choose a correct communication type")
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every round
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            #print("Acc:", acc)
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))


        # print global training loss after every 'i' rounds
        #if (round+1) % print_every == 0:
        print(f' \nAvg Training Stats after {round + 1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(test_acc))

    #Saving the objects train_loss and train_accuracy
    file_name = 'C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/{}_{}_{}_{}_{}_{}_{}_alpha{}.pkl'.\
       format(args.dataset, args.model, args.communication_rounds, args.num_users, args.frac,
               args.local_ep, args.local_batch_size, args.alpha)

    with open(file_name, 'wb') as f:
       pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/loss_federated.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_batch_size))
    #plt.show()

    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/accuracy_federated.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_batch_size))
    #plt.show()
