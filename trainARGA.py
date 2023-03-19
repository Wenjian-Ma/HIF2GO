from GAE_model import GAE,GraphMAE,ARGMA,Discriminator,GCN_encoder,GCN_decoder
import numpy as np
import torch,dgl
from utils import preprocess_graph,sparse_to_tuple
import scipy.sparse as sp
import torch.nn as nn
from loss import sce_loss,sce_kl_loss

def sparse_to_dense(sparse):
    count = 0
    metrics = np.zeros(sparse[2])
    for index in sparse[0]:
        row = int(index[0])
        col = int(index[1])
        metrics[row][col] = sparse[1][count]
        count = count + 1
    return metrics

def train_ARGA(features, adj_train, args, graph_type,device):
    if graph_type == 'ppi':
        epoch = args.epochs_ppi
    elif graph_type == 'sequence_similarity':
        epoch = args.epochs_simi

    # adj_orig = adj_train
    # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # adj_orig.eliminate_zeros()

    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_norm_dense = torch.Tensor(sparse_to_dense(adj_norm)).cuda(device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label_dense = torch.Tensor(sparse_to_dense(adj_label)).cuda(device)
    feature = torch.Tensor(features.toarray()).cuda(device)

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    if args.model == 'GraphMAE':
        model = GraphMAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero, hidden1=args.hidden1, hidden2=args.hidden2,device=device,mask_rate=args.mask_rate,enc_type=args.enc_type,dec_type=args.dec_type)
    elif args.model == 'ARGMA':
        encoder = GCN_encoder(num_features, num_nodes, features_nonzero, args.hidden1, args.hidden2,device, dropout=0.0)
        decoder = GCN_decoder(num_features, num_nodes, features_nonzero, args.hidden1,args.hidden2, device, dropout=0.0)
        discriminator = Discriminator(args.hidden2, args.hidden1, 1)
        model = ARGMA(encoder,decoder,discriminator,num_nodes=num_nodes,num_features=num_features,mask_rate=args.mask_rate)
    else:
        raise Exception('The model '+ args.model+' are not supported!')

    if args.enc_type != 'GCN':
        adj_norm_dense = sp.csr_matrix(adj_norm_dense.cpu().numpy())
        adj_norm_dense = dgl.DGLGraph(adj_norm_dense).to(device)

    model = model.cuda(device)
    # optimizer setting
    if args.model == 'ARGMA':
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(),lr=0.001)
        decoder_optimizer = torch.optim.Adam(model.decoder.parameters(),lr=0.001)
        kit_optimizer = torch.optim.Adam(model.encoder_to_decoder.parameters(),lr=0.001)
        mask_optimizer = torch.optim.Adam([{'params':model.enc_mask_token}],lr=0.001)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=0.0002)
    else:
        model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    BCELoss = nn.BCEWithLogitsLoss()

    for e in range(epoch):
        model.train()
        if args.model == 'ARGMA':
            encoder_optimizer.zero_grad()
            kit_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            mask_optimizer.zero_grad()
        else:
            model_optimizer.zero_grad()
        if args.model == 'GAE':

            z,output = model(adj_norm_dense,feature)

            loss = BCELoss(torch.reshape(output,[1,-1]),torch.reshape(adj_label_dense,[1,-1]))

        elif args.model == 'GraphMAE':

            x_init, x_rec,z,z_with_mask,adj_rec = model(adj_norm_dense,feature)

            loss = sce_loss(x_rec,x_init,alpha=args.alpha)
            ##
            # loss2 = BCELoss(torch.reshape(adj_rec,[1,-1]),torch.reshape(adj_label_dense,[1,-1]))
            # loss = loss + loss2
            ##

        elif args.model == 'ARGMA':
            adj, mask_x, (mask_nodes, keep_nodes) = model.encoding_mask_noise(adj_norm_dense,feature,args.mask_rate)
            z_with_mask = model.encode(adj_norm_dense, mask_x)
            for i in range(5):
                discriminator.train()
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(z_with_mask)
                discriminator_loss.backward()
                discriminator_optimizer.step()
            rep = model.encoder_to_decoder(z_with_mask)
            rep[mask_nodes] = 0
            x_rec = model.decode(adj_norm_dense,rep)
            loss = sce_loss(x_rec[mask_nodes],feature[mask_nodes])
            # loss2 = model.reg_loss(z_with_mask)
            # loss = loss + loss2
            z = model.encode(adj_norm_dense,feature)
        loss.backward()

        if args.model == 'ARGMA':
            encoder_optimizer.step()
            decoder_optimizer.step()
            kit_optimizer.step()
            mask_optimizer.step()
        else:
            model_optimizer.step()

        print('Epoch '+str(e)+':\t'+str(loss.item()))

    return z
