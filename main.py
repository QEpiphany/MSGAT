#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import Data, split_validation
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: /Tmall/Nowplaying/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size') #64,100,256,512
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay 3')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--w_ne', type=float, default=1.7, help='neighbor weight') #digi：1.7 Tmall 0.9
parser.add_argument('--gama', type=float, default=1.7, help='cos_sim') #digi：1.7
parser.add_argument('--num_attention_heads', type=int, default=5, help='Multi-Att heads')

opt = parser.parse_args()
print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.cuda.set_device(0)

def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    elif opt.dataset == 'Tmall':
        n_node = 40728
    elif opt.dataset == 'RetailRocket':
        n_node = 36969
    else:
        n_node = 310

    start = time.time()
    model = trans_to_cuda(SessionGraph(opt, n_node))


    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
        best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

    # Save Model
    PATH = "./final_model/" + opt.dataset + "_model.pkl"
    torch.save(model, PATH)


    # # Load the model for testing
    # PATH = "./final_model/" + opt.dataset + "_model.pkl"
    # model = torch.load(PATH)
    # hit, mrr = test(model, test_data)
    #
    # print('Result:')
    # print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
    # print('-------------------------------------------------------')
    # end = time.time()
    # print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
