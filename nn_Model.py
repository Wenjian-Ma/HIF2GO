import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import torch.nn as nn
import torch

from torch.nn.parameter import Parameter
class nnModel(nn.Module):

    def __init__(self,num_labels,dropout,device,args):
        super(nnModel,self).__init__()

        self.dense1 = nn.Linear(1200,1024)#1312
        self.dense2 = nn.Linear(1024,512)
        self.dense3 = nn.Linear(512, 256)

        self.dense4 = nn.Linear(256,num_labels)#num_labels

        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.LeakyReLU()

        self.device = device

        self.dropout = nn.Dropout(dropout)

        ########################################
        self.dense_lm1 = nn.Linear(1280,512)

        self.dense_lm2 = nn.Linear(512,400)#512
        ##################correlation#######################
        self.emb1 = nn.Linear(num_labels,512)
        self.emb2 = nn.Linear(512, num_labels)
        self.num_labels = num_labels
        self.bn4 = nn.BatchNorm1d(200)
        self.bn5 = nn.BatchNorm1d(512)



    def forward(self,emb,lm):
        lm = self.dense_lm1(lm)
        lm = self.relu(lm)
        lm = self.dropout(lm)
        lm = self.dense_lm2(lm)
        lm = self.relu(lm)
        lm = self.dropout(lm)
        #############
        mean = (emb[:,:400]+emb[:,400:]+lm)//3
        #############
        emb = torch.cat((emb[:,:400]+mean,emb[:,400:]+mean,lm+mean),1)

        #emb = torch.cat((emb[:,:400],lm),1)

        output = self.dense1(emb)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense3(output)
        output = self.bn3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.dense4(output)

        # output = self.bn4(output)
        # output = self.relu(output)
        # output = self.dropout(output)


        # go_embeddings = self.GCN1(adj,go_embeddings)
        # go_embeddings = self.relu(go_embeddings)
        # go_embeddings = self.GCN2(adj,go_embeddings)
        # go_embeddings = self.relu(go_embeddings)
        # go_embeddings = torch.matmul(output, go_embeddings.transpose(0,1))#.transpose(0,1)
        # #
        # # #go_embeddings = torch.cat((output,go_embeddings),1)
        # #
        # go_embeddings = self.emb1(go_embeddings)
        # # go_embeddings = self.bn5(go_embeddings)
        # go_embeddings = self.relu(go_embeddings)
        # go_embeddings = self.dropout(go_embeddings)
        #
        # go_embeddings = self.emb2(go_embeddings)

        #output = torch.matmul(output,go_embeddings.t())

        output = self.sigmoid(output)

        return output


