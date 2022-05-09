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
from models.deterministic import MF, NeuMF, DiverseMF, NeuMF_cvae, PFD, PFD_align
from my_utils import make_model_path, Logger, ms2str
import settings

import numpy as np
from sklearn.metrics import roc_auc_score


def get_ranking_loss(args, batch_data, model, lossFun):
    # get input and target and forward
    slates = torch.LongTensor(batch_data["slates"]).to(model.device)
    users = torch.LongTensor(batch_data["users"]).to(model.device)
    targets = batch_data["responses"].clone().detach().to(torch.float).to(model.device)
#     targets = torch.tensor(batch_data["responses"]).to(torch.float).to(model.device)

    # loss
    pred, stu_out, tea_out = model.forward(slates, targets, u = users)
    prob = model.m(pred)
    loss = lossFun(model.m(pred), targets)
    
    if args.model == 'neumf_cvae':
        stu_out = torch.cat(stu_out,1)
        tea_out = torch.cat(tea_out,1)
        mse = nn.MSELoss()
        loss += args.mse_weight * mse(stu_out, tea_out)
    elif args.model == 'pfd':
        tea_out_array = []
        for i in range(len(tea_out)):
            tea_out_array.append(tea_out[i].unsqueeze(-1))
            
        tea_out = torch.cat(tea_out_array, 1)
        loss += lossFun(model.m(tea_out), targets)
        #prob_stu = model.m(pred).detach()
        #prob_tea = model.m(stu_out)
        with torch.no_grad():
            tea_out_id =  tea_out.clone().detach() 
        mse = nn.MSELoss()
        loss += args.mse_weight * mse(tea_out_id, pred)
    return loss

def get_ranking_AUC(batch_data, model):
    # get input and target and forward
    slates = torch.LongTensor(batch_data["slates"]).to(model.device)
    users = torch.LongTensor(batch_data["users"]).to(model.device)
    
    targets = batch_data["responses"].clone().detach().to(torch.float).to(model.device)
#     targets = torch.tensor(batch_data["responses"]).to(torch.float).to(model.device)

    # loss
    pred, stu_out, tea_out = model.forward(slates, targets, u = users)
    prob = model.m(pred)
    
    return prob, targets



def train_ranking(trainset, valset, model, model_path, logger, resp_model, \
                    bs, epochs, lr, decay, early_stop, testset, cross):
    '''
    @input:
    - trainset and valset: data_loader.UserSlateResponseDataset
    - f_size: embedding size for item and user
    - s_size: slate size
    - model: generative model (list_cvae_with_prior, slate_cvae)
    - bs: batch size
    - epochs: number of epoch
    - lr: learning rate
    - decay: weight decay
    - early_stop
    '''
    
    logger.log("----------------------------------------")
    logger.log("Train user response model as simulator")
    logger.log("\tbatch size: " + str(bs))
    logger.log("\tnumber of epoch: " + str(epochs))
    logger.log("\tlearning rate: " + str(lr))
    logger.log("\tweight decay: " + str(decay))
    logger.log("----------------------------------------")
    model.log(logger)
    logger.log("----------------------------------------")

    # data loaders
    trainLoader = DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 0)
    valLoader = DataLoader(valset, batch_size = bs, shuffle = False, num_workers = 0)    
    testLoader = DataLoader(testset, batch_size = bs, shuffle = False, num_workers = 0)    

    # loss function and optimizer
    BCE = nn.BCELoss()
    m = nn.Sigmoid()
    optimizer = opt.Adam(model.parameters(), lr = lr, weight_decay=decay)
    
    runningLoss = []  # step loss history
    trainHistory = [] # epoch training loss
    valHistory = []   # epoch validation loss
    testHistory = []   # epoch test loss
    bestLoss = np.float("inf")
    bestValLoss = np.float("inf")
    # optimization
    temper = 2
    best_auc = 0
    best_test_auc = 0
    best_epoch = 0
    for epoch in range(epochs):
        logger.log("Epoch " + str(epoch + 1))
        # training
        model.train()
        batchLoss = []
        pbar = tqdm(total = len(trainset))
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()
            loss = get_ranking_loss(args, batchData, model, BCE)

            batchLoss.append(loss.item())
            if len(batchLoss) >= 50:
                runningLoss.append(np.mean(batchLoss[-50:]))

            # backward and optimize
            loss.backward()
            optimizer.step()

            # update progress
            pbar.update(len(batchData["users"]))
            
        # record epoch loss
        trainHistory.append(np.mean(batchLoss))
        pbar.close()
#         logger.log("train loss: " + str(trainHistory[-1]))

        model.eval()
        # validation
        batchLoss = []
        batchProb = []
        batchTargets = []
        with torch.no_grad():
            pbar = tqdm(total = len(valset))
            for i, batchData in enumerate(valLoader):
                loss = get_ranking_loss(args, batchData, model, BCE)
                # prob: [b, 5]
                # target: [b, 5]
                prob, targets = get_ranking_AUC(batchData, model)
                prob_new = prob.reshape([-1,1])
                targets_new = targets.reshape([-1,1])
                batchProb.append(prob_new)
                batchTargets.append(targets_new)
                batchLoss.append(loss.item())
                pbar.update(len(batchData["users"]))
            pbar.close()
        pred = torch.cat(batchProb, dim=0).cpu()
        target = torch.cat(batchTargets, dim=0).cpu()    
        val_auc = roc_auc_score(target, pred)
        valHistory.append(np.mean(batchLoss))
#         logger.log("validation Loss: " + str(valHistory[-1]))
        valHistory.append(val_auc)
        
        batchLoss = []
        batchProb = []
        batchTargets = []
        with torch.no_grad():
            pbar = tqdm(total = len(valset))
            for i, batchData in enumerate(testLoader):
                loss = get_ranking_loss(args, batchData, model, BCE)
                # prob: [b, 5]
                # target: [b, 5]
                prob, targets = get_ranking_AUC(batchData, model)
                prob_new = prob.reshape([-1,1])
                targets_new = targets.reshape([-1,1])
                batchProb.append(prob_new)
                batchTargets.append(targets_new)
                batchLoss.append(loss.item())
                pbar.update(len(batchData["users"]))
            pbar.close()
        
        pred = torch.cat(batchProb, dim=0).cpu()
        target = torch.cat(batchTargets, dim=0).cpu()    
        test_auc = roc_auc_score(target, pred)
        testHistory.append(np.mean(batchLoss))
#         logger.log("validation Loss: " + str(valHistory[-1]))
        testHistory.append(test_auc)
        
        logger.log("validation Auc: " + str(valHistory[-1]))
        logger.log("test Auc: " + str(testHistory[-1]))
        if valHistory[-1] > best_auc:
            best_auc = valHistory[-1]
            best_test_auc = testHistory[-1]
            best_epoch = epoch
        
#         # recommendation test
#         n_test_trial = 100
#         enc = torch.zeros(5, n_test_trial)
#         maxnc = torch.zeros(5, n_test_trial)
#         minnc = torch.zeros(5, n_test_trial)
#         with torch.no_grad():
#             # repeat for several trails
#             for k in tqdm(range(n_test_trial)):
#                 # sample users for each trail
#                 sampledUsers = sample_users(resp_model, bs)
#                 # test for different input condition/context
#                 context = torch.zeros(bs, 5).to(model.device)
#                 for i in range(5):
#                     # each time set one more target response from 0 to 1
#                     context[:,i] = 1
#                     # recommend should gives slate features of shape (B, L)
#                     rSlates, _ = model.recommend(context, sampledUsers, return_item = True)
#                     resp = m(resp_model(rSlates.view(bs, -1), sampledUsers))
#                     # the expected number of click
#                     nc = torch.sum(resp,dim=1)
#                     enc[i,k] = torch.mean(nc).detach().cpu()
#                     maxnc[i,k] = torch.max(nc).detach().cpu()
#                     minnc[i,k] = torch.min(nc).detach().cpu()
#         for i in range(5):
#             logger.log("Expected response (" + str(i+1) + "): " + \
#                        str(torch.mean(minnc[i]).numpy()) + "; " + \
#                        str(torch.mean(enc[i]).numpy()) + "; " + \
#                        str(torch.mean(maxnc[i]).numpy()))

        if early_stop:
            # save best model and early termination
            if epoch == 0 or valHistory[-1] < bestValLoss - 1e-3:
                torch.save(model, open(model_path, 'wb'))
                logger.log("Save best model")
                temper = 5
                bestValLoss = valHistory[-1]
            else:
                temper -= 1
                logger.log("Temper down to " + str(temper))
                if temper == 0:
                    logger.log("Out of temper, early termination.")
                    break
        else:
            # save best model and no early termination
            if epoch == 0 or valHistory[-1] < bestValLoss - 1e-3:
                torch.save(model, open(model_path, 'wb'))
#                 logger.log("Save best model")
     
    print("===================================")
    print("cross:", cross)
    print("best_auc: ", best_auc)
    print("best_epoch: ", best_epoch)
    print("===================================")
    logger.log("Move model to cpu before saving")
    bestModel = torch.load(open(model_path, 'rb'))
    bestModel.to("cpu")
    bestModel.device = "cpu"
    torch.save(bestModel, open(model_path, 'wb'))
    return best_auc, best_test_auc

def get_ranking_model(args, response_model):
    if args.model == "mf":
        model = MF(response_model.docEmbed, response_model.userEmbed, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == "diverse_mf":
        model = DiverseMF(response_model.docEmbed, response_model.userEmbed, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == "neumf":
        mlpStruct = [int(v) for v in args.struct[1:-1].split(",")]
        model = NeuMF(response_model.docEmbed, response_model.userEmbed, mlpStruct, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == 'neumf_cvae':
        mlpStruct = [int(v) for v in args.struct[1:-1].split(",")]
        model = NeuMF_cvae(response_model.docEmbed, response_model.userEmbed, mlpStruct, \
                   args.s, args.dim, args.device, fine_tune = True)
    elif args.model == 'pfd':
        mlpStruct = [int(v) for v in args.struct[1:-1].split(",")]
        model = PFD_align(response_model.docEmbed, response_model.userEmbed, mlpStruct, \
                   args.s, args.dim, args.device, fine_tune = True)
    else:
        raise NotImplemented
    return model

def main(args):
    assert not args.nouser
    logPath = make_model_path(args, "log/")
    logger = Logger(logPath)
    if args.dataset != "spotify" and args.dataset != "yoochoose" and args.dataset != "movielens":
#         args.sim_root = True
        respModel, trainset, valset = dae.load_simulation(args, logger)
    elif args.dataset == "movielens":
        bestauc_set = []
        bestauc_val_set = []
        # train cross num
        for i in range(1,11):
            train, val, test = dae.read_movielens(entire = False, cross = 1)
            trainset = UserSlateResponseDataset(train["features"], train["sessions"], train["responses"], args.nouser)
            valset = UserSlateResponseDataset(val["features"], val["sessions"], val["responses"], args.nouser)
            testset = UserSlateResponseDataset(test["features"], test["sessions"], test["responses"], args.nouser)
            respModel = torch.load(open(args.resp_path, 'rb'))

    #     # do sampling softmax
    #     trainset.init_sampling(args.nneg)
    #     valset.init_sampling(args.nneg)

            respModel.to(args.device)
            respModel.device = args.device

            # generative model
            gen_model = get_ranking_model(args, respModel)
            gen_model.to(args.device)

            modelPath = make_model_path(args, "model/")
            best_val_auc, best_test_auc = train_ranking(trainset, valset, gen_model, modelPath, logger, respModel, \
                        args.batch_size, args.epochs, args.lr, args.wdecay, args.early_stop, testset, i)
            bestauc_set.append(best_test_auc)
            bestauc_val_set.append(best_val_auc)
        print('test auc:')
        print(bestauc_set)
        print(np.mean(bestauc_set))
        print('val auc:')
        print(bestauc_val_set)
        print(np.mean(bestauc_val_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='spotify', help='dataset keyword from ' + str(dae.DATA_KEYS))
#     parser.add_argument('--sim_root', type=bool, default=True, help="load data")
    parser.add_argument('--dim', type=int, default=128, help='number of latent features')
    parser.add_argument('--sim_dim', type=int, default=128, help='number of latent features')
    parser.add_argument('--s', type=int, default=5, help='number of items in a slate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--model', type=str, default='NeuMF_cvae', help='model keyword')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/cuda:0/...')
    parser.add_argument('--nneg', type=int, default=1000, help='number of negative samples for softmax during training')
    parser.add_argument('--nouser', action='store_true', help='user may or may not be considered as input, make sure to change the corresponding model structure and environment')
    parser.add_argument('--mse_weight', type=float, default=0.5, help='mse_weight')
#     parser.add_argument('--balance', action='store_true', help='apply response balancing')
    
    # used by NeuMF models
    parser.add_argument('--struct', type=str, default="[256,256,1]", help='mlp structure for prediction')
    
    # if training generative model
    parser.add_argument('--response', action='store_true', help='training response model for the generation model')
    parser.add_argument('--resp_path', type=str, default="resp/resp_[48,256,256,5]_spotify_BS64_dim8_lr0.00030_decay0.00010", help='trained user response model, only valid when training generative rec model')
    parser.add_argument('--early_stop', type=int, default=1, help='early stop')
    
    # used when simulation
    parser = dae.add_sim_parse(parser)
    
    args = parser.parse_args()
    main(args)
        