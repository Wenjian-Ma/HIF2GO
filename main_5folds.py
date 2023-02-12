import argparse
from input_data import load_data,load_labels
from trainARGA import train_ARGA
from trainNN import train_nn
import numpy as np
import pandas as pd
import os,sys
import torch
from preprocessing import PFPDataset,collate
from torch.utils.data import DataLoader
import warnings

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):

    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:'+args.device)

    embeddings_list = []
    for graph in args.graphs:
        print("#############################")
        print("Training",graph)
        adj, features = load_data(graph, uniprot, args)
        embeddings = train_ARGA(features, adj, args, graph,device)
        embeddings_list.append(embeddings.cpu().detach().numpy())
    embeddings = np.hstack(embeddings_list)

    #embeddings = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/embeddings.npy')
    np.random.seed(5959)
    #np.save('/home/sgzhang/perl5/GAT-GO/Structure2GO-label/data/embeddings.npy', embeddings)
    cc, mf, bp = load_labels(uniprot)

    # cc[cc==0] = -1
    # mf[mf==0] = -1
    # bp[bp==0] = -1

    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test
    all_idx = list(range(cc.shape[0]))
    np.random.shuffle(all_idx)

    train_idx = all_idx[:num_train]
    fold_num = int(np.floor(num_train / 5.))
    valid_folds_idx = [train_idx[:fold_num],train_idx[fold_num:fold_num*2],train_idx[fold_num*2:fold_num*3],train_idx[fold_num*3:fold_num*4],train_idx[fold_num*4:]][args.fold]
    for i in valid_folds_idx:
        train_idx.remove(i)


    ESM = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/ESM-embeddings.npy')

    Y_train_cc = cc[train_idx]
    Y_train_bp = bp[train_idx]
    Y_train_mf = mf[train_idx]

    Y_test_cc = cc[valid_folds_idx]
    Y_test_bp = bp[valid_folds_idx]
    Y_test_mf = mf[valid_folds_idx]

    X_train = embeddings[train_idx]
    X_test = embeddings[valid_folds_idx]
    LM_train = ESM[train_idx]
    LM_test = ESM[valid_folds_idx]
    ##########################


    train_data_cc = PFPDataset(train_data_X=X_train, train_data_Y=Y_train_cc,lm = LM_train)
    train_data_bp = PFPDataset(train_data_X=X_train, train_data_Y=Y_train_bp,lm = LM_train)
    train_data_mf = PFPDataset(train_data_X=X_train, train_data_Y=Y_train_mf,lm = LM_train)

    test_data_cc = PFPDataset(train_data_X=X_test, train_data_Y=Y_test_cc,lm = LM_test)
    test_data_bp = PFPDataset(train_data_X=X_test, train_data_Y=Y_test_bp,lm = LM_test)
    test_data_mf = PFPDataset(train_data_X=X_test, train_data_Y=Y_test_mf,lm = LM_test)

    dataset_train_cc = DataLoader(train_data_cc, batch_size=128, shuffle=True, collate_fn=collate, drop_last=False)
    dataset_train_bp = DataLoader(train_data_bp, batch_size=128, shuffle=True, collate_fn=collate, drop_last=False)
    dataset_train_mf = DataLoader(train_data_mf, batch_size=128, shuffle=True, collate_fn=collate, drop_last=False)

    dataset_test_cc = DataLoader(test_data_cc, batch_size=128, shuffle=False, collate_fn=collate, drop_last=False)
    dataset_test_bp = DataLoader(test_data_bp, batch_size=128, shuffle=False, collate_fn=collate, drop_last=False)
    dataset_test_mf = DataLoader(test_data_mf, batch_size=128, shuffle=False, collate_fn=collate, drop_last=False)

    print("Start running supervised model...")
    # save_path = os.path.join(args.data_path, args.species,"results_new/results_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue))

    print("###################################")
    print('----------------------------------')
    print('MF')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
             train_loader=dataset_train_mf,go=mf, test_loader=dataset_test_mf,term='MF')

    # print("###################################")
    # print('----------------------------------')
    # print('BP')
    #
    # train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
    #          train_loader=dataset_train_bp, go=bp, test_loader=dataset_test_bp,term='BP')
    #
    # print("###################################")
    # print('----------------------------------')
    # print('CC')
    #
    # train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_cc.shape[1],
    #          train_loader=dataset_train_cc, go=cc, test_loader=dataset_test_cc,term='CC')


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=5, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi','sequence_similarity'], help="lists of graphs to use.")#'ppi',
    parser.add_argument('--species', type=str, default="Human", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--device', type=str, default=0, help="cuda device.")
    parser.add_argument('--fold', type=int, default=0, help="5 folds.")
    parser.add_argument('--model', type=str, default='GraphMAE', help="model(GAE/GraphMAE).")
    parser.add_argument('--epochs_ppi', type=int, default=140, help="Number of epochs to train ppi.")#200 for GAE/VGAE   140 for GraphMAE
    parser.add_argument('--epochs_simi', type=int, default=120, help="Number of epochs to train similarity network.")#75 for GAE/VGAE   120 for GraphMAE
    parser.add_argument('--thr_combined', type=float, default=0.0, help="threshold for combiend ppi network.")#0.3
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")# 1e-4
    parser.add_argument('--mask_rate', type=float, default=0.9, help="mask rate.")
    parser.add_argument('--alpha', type=int, default=3, help="alpha for sce_loss.")
    parser.add_argument('--enc_type', type=str, default='GCN', help="encoder_type(GCN/GAT/GIN).")
    parser.add_argument('--dec_type', type=str, default='GCN', help="decoder_type(MLP/GCN/GAT/GIN).")
    args = parser.parse_args()
    print(args)
    train(args)