import argparse
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import time as timer

import data_extract as dae
from data_loader import UserSlateResponseDataset
from env.response_model import UserResponseModel_MLP, sample_users
import my_utils as utils
import settings

#######################################################
#                train response model                 #
#######################################################

def train_response_model(trainset, valset, testset, f_size, s_size, struct, bs, epochs, lr, decay, device, model_path, logger):
    
#     python pretrain_env.py --dataset urmpmr --sim_dim 8 --n_user 1000 --n_item 3000 --n_train 100000 --n_val 10000 --n_test 10000 --pbias_min=-0.2 --pbias_max 0.2 --mr_factor 0.2 --dim 8 --resp_struct [48,256,256,5] --batch_size 64 --lr 0.001 --wdecay 0.001 --device cuda:0
        
    '''
    @input:
    - trainset and valset: data_loader.UserSlateResponseDataset
    - f_size: embedding size for item and user
    - s_size: slate size
    - struct: response model structure
    - bs: batch size
    - epochs: number of epoch
    - lr: learning rate
    - decay: L2 norm coefficient
    - device: "cpu", "cuda:0", etc.
    '''
    logger.log("Train user response model as simulator")
    logger.log("\tfeature size: " + str(f_size))
    logger.log("\tslate size: " + str(s_size))
    logger.log("\tstruct: " + str(struct))
    logger.log("\tbatch size: " + str(bs))
    logger.log("\tnumber of epoch: " + str(epochs))
    logger.log("\tlearning rate: " + str(lr))
    logger.log("\tdevice: " + device)
    
    # set up model
    max_iid = max(trainset.max_iid, max(testset.max_iid, valset.max_iid))
    max_uid = max(trainset.max_uid, max(testset.max_uid, valset.max_uid))
    print('max_iid: %d, max_uid: %d' % (max_iid, max_uid))
    model = UserResponseModel_MLP(max_iid, max_uid, \
                                  f_size, s_size, struct, device, trainset.noUser)
    model.to(device)

    # data loaders
    trainLoader = DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 0)
    valLoader = DataLoader(valset, batch_size = bs, shuffle = False, num_workers = 0)

    # loss function and optimizer
    BCE = nn.BCELoss()
    m = nn.Sigmoid()
    optimizer = opt.Adam(model.parameters(), lr = lr, weight_decay = decay)
#     optimizer = opt.SGD(model.parameters(), lr=lr, weight_decay = decay)

    
    runningLoss = []  # step loss history
    trainHistory = [] # epoch training loss
    valHistory = []   # epoch validation loss
    bestLoss = np.float("inf")
    bestValLoss = np.float("inf")
    # optimization
    temper = 3
    for epoch in range(epochs):
        logger.log("Epoch " + str(epoch + 1))
        # training
        batchLoss = []
        pbar = tqdm(total = len(trainset))
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()

            # get input and target and forward
            slates = torch.LongTensor(batchData["slates"]).to(model.device)
            users = torch.LongTensor(batchData["users"]).to(model.device)
            targets = torch.tensor(batchData["responses"]).to(torch.float).to(model.device)
            pred = model.forward(slates, users)

            # loss
            loss = BCE(m(pred.reshape(-1)), targets.reshape(-1))
            batchLoss.append(loss.item())
            if len(batchLoss) >= 50:
                runningLoss.append(np.mean(batchLoss[-50:]))

            # backward and optimize
            loss.backward()
            optimizer.step()

            # update progress
            pbar.update(len(users))
            
        print("Embedding norm: " + str(torch.norm(model.docEmbed.weight[0], p = 2)))

        # record epoch loss
        trainHistory.append(np.mean(batchLoss))
        pbar.close()
        logger.log("train loss: " + str(trainHistory[-1]))

        # validation
        batchLoss = []
        with torch.no_grad():
            for i, batchData in tqdm(enumerate(valLoader)):

                # get input and target and forward
                slates = torch.LongTensor(batchData["slates"]).to(model.device)
                users = torch.LongTensor(batchData["users"]).to(model.device)
                targets = torch.tensor(batchData["responses"]).to(torch.float).to(model.device)
                pred = model.forward(slates, users)

                # loss
                loss = BCE(m(pred.reshape(-1)), targets.reshape(-1))
                batchLoss.append(loss.item())

        valHistory.append(np.mean(batchLoss))
        logger.log("Validation Loss: " + str(valHistory[-1]))

        # save best model and early termination
        if epoch == 0 or valHistory[-1] < bestValLoss - 1e-4:
            torch.save(model, open(model_path, 'wb'))
            logger.log("Save best model")
            temper = 3
            bestValLoss = valHistory[-1]
        else:
            temper -= 1
            logger.log("Temper down to " + str(temper))
            if temper == 0:
                logger.log("Out of temper, early termination.")
                break
                
    logger.log("Move model to cpu before saving")
    bestModel = torch.load(open(model_path, 'rb'))
    bestModel.to("cpu")
    bestModel.device = "cpu"
    torch.save(bestModel, open(model_path, 'wb'))

#######################################
#                main                 #
#######################################

def main(args):
    logPath = utils.make_resp_model_path(args, "log/")
    logger = utils.Logger(logPath)
    
    # 模拟的数据，load进来模拟的数据
    if args.dataset != "yoochoose" and args.dataset != "movielens": # simulation envirionment
        respModel, trainset, valset = dae.load_simulation(args, logger)
    else: # real-world datasets
        if args.dataset == "yoochoose":
            train, val, test = dae.read_yoochoose(entire_set = True)
            args.nouser == True
            trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
            trainset.balance_n_click()
            valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
        elif args.dataset == "movielens":
            train, val, test = dae.read_movielens(entire = False)
            trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
            valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
            testset = UserSlateResponseDataset(test["features"], test["sessions"], test["responses"], args.nouser)
    # train response model
    modelPath = utils.make_resp_model_path(args, "resp/")
    # [48,256,256,5]
    # args.resp_struct[1:-1] --
    struct = [int(v) for v in args.resp_struct[1:-1].split(",")]
    import setproctitle 
    setproctitle.setproctitle("Kassandra")
    train_response_model(trainset, valset, testset,\
                         args.dim, args.s, struct, args.batch_size, \
                         args.epochs, args.lr, args.wdecay, args.device, modelPath, logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # real-world dataset configuration
    parser = dae.add_data_parse(parser)
    # simulation configuration
    parser = dae.add_sim_parse(parser)
    # training configuration
    parser = utils.add_training_parse(parser)
    # response model configuration
    parser.add_argument('--dim', type=int, default=8, help='item/user embedding size')
    parser.add_argument('--resp_struct', type=str, default="[48,256,256,5]", help='mlp structure for prediction')
    
    args = parser.parse_args()

    main(args)
        
