import os
import scipy.sparse as sp
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import torch
import torch.nn as nn
from torch import optim
from nn_Model import nnModel
from evaluation import get_results
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from utils import preprocess_graph
from loss import BCEFocalLosswithLogits

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def sparse_to_dense(sparse):
    count = 0
    metrics = np.zeros(sparse[2])
    for index in sparse[0]:
        row = int(index[0])
        col = int(index[1])
        metrics[row][col] = sparse[1][count]
        count = count + 1
    return metrics

def train_nn(args,train_loader,device,input_dim,output_dim,go,test_loader,term):
    #######################Correlation Matrix of GO terms############################
    # print('Processing Correlation matrix of GO terms...')
    # if term == 'MF':
    #     relation_matrix = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/mf_relation_matrix_plus.npy')
    #     go_embeddings = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/GO_list/mf_go_embeddings.npy')
    #     go_embeddings = torch.Tensor(go_embeddings).cuda(device)
    # elif term == 'BP':
    #     relation_matrix = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/bp_relation_matrix_plus.npy')
    #     go_embeddings = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/GO_list/bp_go_embeddings.npy')
    #     go_embeddings = torch.Tensor(go_embeddings).cuda(device)
    # elif term == 'CC':
    #     relation_matrix = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/cc_relation_matrix_plus.npy')
    #     go_embeddings = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/GO_list/cc_go_embeddings.npy')
    #     go_embeddings = torch.Tensor(go_embeddings).cuda(device)
    #
    # relation_matrix = (relation_matrix >= args.threshold_t) + 0.0
    # relation_matrix = relation_matrix * args.threshold_p / (relation_matrix.sum(0, keepdims=True) + 1e-6)
    # relation_matrix = relation_matrix + np.identity(relation_matrix.shape[1], np.int) * (1 - args.threshold_p)
    # relation_matrix = gen_adj(torch.from_numpy(relation_matrix).float()).cuda(device)

    ####################################################

    Epoch = 200

    model = nnModel(output_dim,dropout=0.3,device=device,args=args)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 128 0.0005

    bceloss = nn.BCELoss()

    #bceloss = BCEFocalLosswithLogits()
    for e in range(Epoch):
        model.train()
        for batch_idx,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            optimizer.zero_grad()
            Y_label = batch[1].to(device)
            emb = batch[0].to(device)
            lm = batch[2].to(device)

            output = model(emb,lm)
            loss_out = bceloss(output, Y_label)
            loss_out.backward()
            optimizer.step()

        model.eval()
        total_preds = torch.Tensor().to(device)
        total_labels = torch.Tensor().to(device)
        with torch.no_grad():
            for batch_test_idx,batch_test in enumerate(tqdm(test_loader,mininterval=0.5,desc='Testing',leave=False,ncols=50)):
                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_test = batch_test[2].to(device)
                output_test = model(emb_test,lm_test)
                total_preds = torch.cat((total_preds, output_test), 0)
                total_labels = torch.cat((total_labels, label_test), 0)


            loss_test = bceloss(total_preds,total_labels)


        perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())

        # torch.save(model.state_dict(),'/home/sgzhang/perl5/GAT-GO/HIF2GO/model/Human/model'+str(perf['all']['M-aupr'])+'.pkl')

        print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss_out.item(), '\tTest loss:\t',
              loss_test.item(), '\n\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],
              '\tF-max:\t', perf['all']['F-max'])
        # if e == 199:
        #     torch.save(total_preds,'/home/sgzhang/perl5/GAT-GO/HIF2GO/data/backup/pred(no-lm).pt')
        #
        #     exit()
