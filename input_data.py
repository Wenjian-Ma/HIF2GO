import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import scale
import os
from tqdm import tqdm

def load_ppi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        if float(temp[2]) >= thr:
            #adj[int(temp[0]), int(temp[1])] = 1
            adj[int(temp[0]), int(temp[1])] = float(temp[2])
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
    return adj

def load_simi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether evalue smaller than the threshold
        if float(temp[2]) <= thr:
            adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T

    return adj


def load_labels(uniprot):
    print('loading labels...')
    # load labels (GO)
    cc = uniprot['cc_label'].values
    cc = np.hstack(cc).reshape((len(cc), len(cc[0])))

    bp = uniprot['bp_label'].values
    bp = np.hstack(bp).reshape((len(bp), len(bp[0])))

    mf = uniprot['mf_label'].values
    mf = np.hstack(mf).reshape((len(mf), len(mf[0])))

    return cc, mf, bp


def load_data(graph_type, uniprot, args):
    print('loading data...')

    def reshape(features):
        return np.hstack(features).reshape((len(features), len(features[0])))

    # get feature representations
    features_seq = scale(reshape(uniprot['CT'].values))
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)
    # features_disease = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/disease_10.npy')
    # features_tissue = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/tissue_10.npy')
    features_pathway = np.load('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/'+args.species+'/pathway.npy')
    print('generating features...')
    # attribute=6
    if graph_type == "ppi":
        attribute = args.ppi_attributes
    elif graph_type == "sequence_similarity":
        attribute = args.simi_attributes

    if attribute == 0:
        features = np.identity(uniprot.shape[0])
        print("Without features")
    elif attribute == 1:
        features = features_seq
        print("Only use sequence feature")
    elif attribute == 2:
        features = features_loc
        print("Only use location feature")
    elif attribute == 3:
        features = features_domain
        print("Only use domain feature")
    elif attribute == 5:
        features = np.concatenate((features_loc, features_domain,features_pathway), axis=1)
        print("use location and domain features")
    elif attribute == 6:#,features_disease,features_tissue,features_pathway
        features = np.concatenate((features_seq, features_loc, features_domain,features_pathway), axis=1)
        print("Use all the features")
    elif attribute == 7:
        features = np.concatenate((features_seq, features_loc, features_domain, np.identity(uniprot.shape[0])), axis=1)
        print("Use all features plus identity")

    features = sp.csr_matrix(features)

    print('loading graph...')
    if graph_type == "sequence_similarity":
        filename = os.path.join(args.data_path, args.species, "sequence_similarity.txt")
        adj = load_simi_network(filename, uniprot.shape[0], args.thr_evalue)
    elif graph_type == "ppi":
        filename = os.path.join(args.data_path, args.species, "ppi.txt")
        adj = load_ppi_network(filename, uniprot.shape[0], args.thr_combined)

    adj = sp.csr_matrix(adj)

    return adj, features
