import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from heapq import heappush, heappop
from tqdm import tqdm
import hyperparams

class RankModel(nn.Module):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True):
        """
        @input:
        - embeddings: pretrained item embeddings
        - u_embeddings: pretrained user embeddings
        - slate_size: number of items in a slate
        - no_user: true if user embeddings are ignored during training/inference
        - device: cpu/cuda:x
        - fine_tune: true if want to fine tuning item/user embedding
        """
        super(RankModel, self).__init__()
        print(embeddings.weight.shape[1])
        print(feature_size)
        assert embeddings.weight.shape[1] == feature_size
        assert u_embeddings.weight.shape[1] == feature_size
        self.candidateFlag = False
        self.slate_size = slate_size
        self.feature_size = feature_size
        self.device = device
        print("\tdevice: " + str(self.device))
        
        with torch.no_grad():
            # doc embedding
            print("\tLoad pretrained document latent embedding")
            self.docEmbed = nn.Embedding(embeddings.weight.shape[0], embeddings.weight.shape[1])
            self.docEmbed.weight.data.copy_(F.normalize(embeddings.weight, p = 2, dim = 1))
    #         self.docEmbed.weight.data.copy_(embeddings.weight)
            self.docEmbed.weight.requires_grad=fine_tune
            print("\t\tDoc embedding shape: " + str(self.docEmbed.weight.shape))

            # user embedding
            print("\tCopying user latent embedding")
            self.userEmbed = nn.Embedding(u_embeddings.weight.shape[0], u_embeddings.weight.shape[1])
            self.userEmbed.weight.data.copy_(F.normalize(u_embeddings.weight, p = 2, dim = 1))
#             self.userEmbed.weight.data.copy_(u_embeddings.weight)
            self.userEmbed.weight.requires_grad=fine_tune
            print("\t\tUser embedding shape: " + str(self.userEmbed.weight.shape))
            
        self.m = nn.Sigmoid()
    
    def point_forward(self, users, items, slate):
        raise NotImplemented
    
    def forward(self, s, r, candidates = None, u = None):
        pred = torch.zeros_like(s).to(torch.float)
        stu_set  = []
        tea_set = []
#         pred1 = torch.zeros_like(s).to(torch.float)
        for i in range(self.slate_size):
            pred[:,i], stu_out, tea_out = self.point_forward(u, s[:,i], s)
            stu_set.append(stu_out)
            tea_set.append(tea_out)
        return pred, stu_set, tea_set
        
    def recommend(self, r, u = None, return_item = False):
        raise NotImplemented
    
    def get_recommended_item(self, embeddings):
        candidateEmb = self.docEmbed.weight.data.view((-1,self.feature_size))
        p = torch.mm(embeddings, candidateEmb.t())
        values, indices = torch.max(p,1)
        return indices
        
    def log(self, logger):
        logger.log("\tfeature size: " + str(self.feature_size))
        logger.log("\tslate size: " + str(self.slate_size))
        logger.log("\tdevice: " + str(self.device))

class MF(RankModel):
    """
    Biased MF
    """
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize MF framework...")
        super(MF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        
        print("Initialize user and item biases")
        # user bias
        self.userBias = nn.Embedding(self.userEmbed.weight.shape[0], 1)
        self.userBias.weight.data.fill_(0.001)
        
        # item bias
        self.docBias = nn.Embedding(self.docEmbed.weight.shape[0], 1)
        self.docBias.weight.data.fill_(0.001)
        
        print("Done.")
        
    def point_forward(self, users, items, slate):
        """
        forward pass of MF
        """
        # extract latent embeddings
        uE = self.userEmbed(users.view(-1))
        uB = self.userBias(users.view(-1))
        iE = self.docEmbed(items.view(-1))
        iB = self.docBias(items.view(-1))

        # positive example
        output = torch.mul(uE,iE)\
                            .sum(1).view(-1,1)
        output = output + uB + iB
        return output.view(-1)
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i] = self.point_forward(u[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None

class DiverseMF(MF):
    def __init__(self, embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = True, diversity_alpha = 0.5):
        print("Initialize MF framework...")
        super(DiverseMF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        self.alpha = diversity_alpha
    
    def item_similarity(self, items):
        iE = self.docEmbed(items.view(-1))
        iB = self.docBias(items.view(-1))
        
        item_sim,_ = torch.max(torch.matmul(iE,iE.transpose(0,1)),1)       
        return item_sim
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i] = self.point_forward(u[i], candItems)
        
        item_sim = self.item_similarity(candItems)
        mmr = self.alpha*p - (1-self.alpha)*item_sim
        
        _, recItems = torch.topk(mmr, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None


class NeuMF(RankModel):
    """
    Biased MF
    """
    def __init__(self, embeddings, u_embeddings, struct, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize NeuMF model...")
        super(NeuMF, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        # MLP structure hyperparams
        self.structure = struct
        assert len(self.structure) >= 2
#         assert self.structure[0] == 2 * feature_size
        
        # GMF part
        self.gmf = nn.Linear(feature_size, 1)
        
        print("\tCreating embedding")
        # MLP user embedding
        self.userMLPEmbed = nn.Embedding(u_embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.userMLPEmbed.weight)
        # MLP item embedding
        self.docMLPEmbed = nn.Embedding(embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.docMLPEmbed.weight)

        # setup model structure
        print("\tCreating predictive model")
        # mlp part
        self.mlp_modules = list()
        for i in range(len(self.structure) - 1):
            module = nn.Linear(self.structure[i], self.structure[i+1])
            torch.nn.init.kaiming_uniform_(module.weight)
            self.mlp_modules.append(module)
            self.add_module("mlp_" + str(i), module)
        # final output
        self.outAlpha = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad = True)

    def point_forward(self, users, items, slate):
        # user embeddings
        uME = self.userMLPEmbed(users.view(-1))
        uGE = self.userEmbed(users.view(-1))
        # item embeddings
        iME = self.docMLPEmbed(items.view(-1))
        iGE = self.docEmbed(items.view(-1))

        # GMF output
        gmfOutput = self.gmf(torch.mul(uGE, iGE))
        # MLP output
        X = torch.cat((uME, iME), 1)

        for i in range(len(self.mlp_modules) - 1):
            layer = self.mlp_modules[i]
            X = F.relu(layer(X))
            
        mlpOutput = self.mlp_modules[-1](X)
        # Integrate GMF and MLP outputs
        out = self.outAlpha * gmfOutput + (1 - self.outAlpha) * mlpOutput
        return out.view(-1), 0, 0
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        users = torch.ones(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i], _, _ = self.point_forward(users * u.view(-1)[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None
        
        
class NeuMF_cvae(RankModel):
    """
    Biased MF + cvae
    """
    def __init__(self, embeddings, u_embeddings, struct, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize NeuMF model...")
        super(NeuMF_cvae, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        # MLP structure hyperparams
        self.structure = struct
        assert len(self.structure) >= 2
#         assert self.structure[0] == 2 * feature_size
        
        # GMF part
        self.gmf = nn.Linear(feature_size, 1)
        
        print("\tCreating embedding")
        # MLP user embedding
        self.userMLPEmbed = nn.Embedding(u_embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.userMLPEmbed.weight)
        # MLP item embedding
        self.docMLPEmbed = nn.Embedding(embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.docMLPEmbed.weight)
        
        # set for cvae structure
        self.flush_seq_len = 18
        self.flush_sequence_nn_layers = [16, 8, 16]
        self.conditional_decoder_layers = [2]
        self.prior_encoder_layers = [0, 1]
        self.conditional_encoder_layers = [0, 1]
#         self.flush_activation = relu
#         self.cvae_activation = selu

        # setup model structure
        print("\tCreating predictive model")
        # mlp part
        self.mlp_modules = list()
        for i in range(len(self.structure) - 1):
            module = nn.Linear(self.structure[i], self.structure[i+1])
            torch.nn.init.kaiming_uniform_(module.weight)
            self.mlp_modules.append(module)
            self.add_module("mlp_" + str(i), module)
        # final output
        self.outAlpha = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad = True)

    def point_forward(self, users, items, slate):
        # user embeddings
        uME = self.userMLPEmbed(users.view(-1)) #[b,64]
        uGE = self.userEmbed(users.view(-1)) 
        # item embeddings
        iME = self.docMLPEmbed(items.view(-1)) # [64,8]
        iGE = self.docEmbed(items.view(-1))
        # sltae embeddings
        i1, i2, i3, i4, i5 = slate[:,0], slate[:,1], slate[:,2], slate[:,3], slate[:,4]
        i1ME =  self.docMLPEmbed(i1.view(-1))
        i2ME =  self.docMLPEmbed(i2.view(-1))
        i3ME =  self.docMLPEmbed(i3.view(-1))
        i4ME =  self.docMLPEmbed(i4.view(-1))
        i5ME =  self.docMLPEmbed(i5.view(-1))
        
        slate_emb = torch.cat((i1ME, i2ME, i3ME, i4ME, i5ME), 1) # [64, 40]
        slate_emb = nn.Linear(slate_emb.shape[1], 16).to(self.device)(slate_emb)
        
        # GMF output
        gmfOutput = self.gmf(torch.mul(uGE, iGE))
        # MLP output
        X = torch.cat((uME, iME), 1) 
        # student input
        student_input = uME
        for index in self.prior_encoder_layers: # [0, 1]
            node_num = self.flush_sequence_nn_layers[index] # [64, 32]
            student_input = nn.Linear(student_input.shape[1], node_num).to(self.device)(student_input)
            student_input = F.relu(student_input)
        student_out = F.normalize(student_input, p = 2, dim = 1)
        # teacher input
        teacher_input = torch.cat((uME, slate_emb), 1)
        for index in self.conditional_encoder_layers: # [0, 1]
            node_num = self.flush_sequence_nn_layers[index] # [64, 32]
            teacher_input = nn.Linear(teacher_input.shape[1], node_num).to(self.device)(teacher_input)
            teacher_input = F.relu(teacher_input)
        teacher_out =  F.normalize(teacher_input, p = 2, dim = 1)
        
        if self.training:
            cvae_input = torch.cat((student_input, teacher_out), 1)
        else:
            cvae_input = torch.cat((student_input, student_out), 1)
            
        for index in self.conditional_decoder_layers: # [2]
            node_num = self.flush_sequence_nn_layers[index] # 64
            cvae_input = nn.Linear(cvae_input.shape[1], node_num).to(self.device)(cvae_input)
            cvae_input = F.relu(cvae_input)
        cvae_out =  cvae_input
        X = torch.cat((X, cvae_out), 1)
        
        for i in range(len(self.mlp_modules) - 1):
            layer = self.mlp_modules[i]
            X = F.relu(layer(X))
        mlpOutput = self.mlp_modules[-1](X)
        # Integrate GMF and MLP outputs
        out = self.outAlpha * gmfOutput + (1 - self.outAlpha) * mlpOutput
        return out.view(-1), student_out, teacher_out

    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        users = torch.ones(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i], stu_out, tea_out = self.point_forward(users * u.view(-1)[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None
        
class PFD(RankModel):
    """
    Biased MF
    """
    def __init__(self, embeddings, u_embeddings, struct, \
                 slate_size, feature_size, device, fine_tune = True):
        print("Initialize NeuMF model...")
        super(PFD, self).__init__(embeddings, u_embeddings, \
                 slate_size, feature_size, device, fine_tune = fine_tune)
        
        # MLP structure hyperparams
        self.structure = struct
        assert len(self.structure) >= 2
#         assert self.structure[0] == 2 * feature_size
        
        # GMF part
        self.gmf = nn.Linear(feature_size, 1)
        
        print("\tCreating embedding")
        # MLP user embedding
        self.userMLPEmbed = nn.Embedding(u_embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.userMLPEmbed.weight)
        # MLP item embedding
        self.docMLPEmbed = nn.Embedding(embeddings.weight.shape[0], feature_size)
        nn.init.kaiming_uniform_(self.docMLPEmbed.weight)

        # setup model structure
        print("\tCreating predictive model")
        self.mlp_modules = list()
        for i in range(len(self.structure) - 1):
            module = nn.Linear(self.structure[i], self.structure[i+1])
            torch.nn.init.kaiming_uniform_(module.weight)
            self.mlp_modules.append(module)
            self.add_module("mlp_" + str(i), module)
        # teacher mlp part
        self.tea_mlp_modules = list()
        for i in range(len(self.structure) - 1):
            if i == 0:  
                module = nn.Linear(self.structure[i]+16, self.structure[i+1])
            else:
                module = nn.Linear(self.structure[i], self.structure[i+1])
            torch.nn.init.kaiming_uniform_(module.weight)
            self.tea_mlp_modules.append(module)
            self.add_module("tea_mlp_" + str(i), module)
        # final output
        self.outAlpha = nn.Parameter(torch.tensor(0.5).to(self.device), requires_grad = True)

    def point_forward(self, users, items, slate):
        # user embeddings
        uME = self.userMLPEmbed(users.view(-1))
        uGE = self.userEmbed(users.view(-1))
        # item embeddings
        iME = self.docMLPEmbed(items.view(-1))
        iGE = self.docEmbed(items.view(-1))
        
        # sltae embeddings
        i1, i2, i3, i4, i5 = slate[:,0], slate[:,1], slate[:,2], slate[:,3], slate[:,4]
        i1ME =  self.docMLPEmbed(i1.view(-1))
        i2ME =  self.docMLPEmbed(i2.view(-1))
        i3ME =  self.docMLPEmbed(i3.view(-1))
        i4ME =  self.docMLPEmbed(i4.view(-1))
        i5ME =  self.docMLPEmbed(i5.view(-1))
        
        slate_emb = torch.cat((i1ME, i2ME, i3ME, i4ME, i5ME), 1) # [64, 40]
        slate_emb = nn.Linear(slate_emb.shape[1], 16).to(self.device)(slate_emb)
        
    
        # GMF output
        gmfOutput = self.gmf(torch.mul(uGE, iGE))
        # student MLP output
        X = torch.cat((uME, iME), 1)
        for i in range(len(self.mlp_modules) - 1):
            layer = self.mlp_modules[i]
            X = F.relu(layer(X))
        mlpOutput = self.mlp_modules[-1](X)
        # teacher MLP output
        Y = torch.cat((uME, iME, slate_emb), 1) # +16
        for i in range(len(self.tea_mlp_modules) - 1):
            layer = self.tea_mlp_modules[i]
            Y = F.relu(layer(Y))
        teamlpOutput = self.tea_mlp_modules[-1](Y)
        
        # Integrate GMF and MLP outputs
        out = self.outAlpha * gmfOutput + (1 - self.outAlpha) * mlpOutput
        return out.view(-1), teamlpOutput.view(-1), 0
    
    def recommend(self, r, u = None, return_item = False):
        p = torch.zeros(u.shape[0], self.docEmbed.weight.shape[0]).to(self.device)
        candItems = torch.arange(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        users = torch.ones(self.docEmbed.weight.shape[0]).to(torch.long).to(self.device)
        for i in range(u.shape[0]):
            p[i], tea_p[i], _ = self.point_forward(users * u.view(-1)[i], candItems)
        _, recItems = torch.topk(p, self.slate_size)
        if return_item:
            return recItems, None
        else:
            rx = self.docEmbed(recItems).reshape(-1, self.slate_size * self.feature_size)
            return rx, None