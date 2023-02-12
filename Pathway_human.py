from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np

mapping = {}
with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/backup/mapping.txt','r') as f:
    for id,line in enumerate(f):
        if id != 0:
            uid = line.strip().split('\t')[0]
            string_id = line.strip().split('\t')[1]
            mapping[string_id] = uid


######################################
prot_list = []
required_field = ['Disease-gene associations (DISEASES)','Tissue expression (TISSUES)','Reactome Pathways']

prot_info = {}

with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/backup/9606.protein.enrichment.terms.v11.5.txt','r') as f:
    for id,line in enumerate(tqdm(f,total=3584948)):
        if id != 0:
            STRING_id = line.strip().split('\t')[0]
            prot_list.append(STRING_id)
prot_list = list(set(prot_list))
for i in tqdm(prot_list):
    prot_info[i] = {required_field[0]:[],required_field[1]:[],required_field[2]:[]}

with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/backup/9606.protein.enrichment.terms.v11.5.txt','r') as f:
    for id,line in enumerate(tqdm(f,total=3584948)):
        if id != 0:
            STRING_id = line.strip().split('\t')[0]
            field = line.strip().split('\t')[1]
            if field in required_field:
                term = line.strip().split('\t')[2]
                prot_info[STRING_id][field].append(term)

prot_info_new = {}
for i in prot_info.keys():
    if i in mapping.keys():
        prot_info_new[mapping[i]] = prot_info[i]
#################################################################上边不要动
Diseases = []
Tissue = []
Pathway = []
for i in prot_info_new.keys():
    Diseases.extend(prot_info_new[i]['Disease-gene associations (DISEASES)'])
    Tissue.extend(prot_info_new[i]['Tissue expression (TISSUES)'])
    Pathway.extend(prot_info_new[i]['Reactome Pathways'])
Diseases_count = Counter(Diseases)#4167
Tissue_count = Counter(Tissue)#3115
Pathway_count = Counter(Pathway)#2167

def filter(x,threshold):#根据阈值筛选
    count = 0
    term = []
    x = dict(x)
    for i in x.keys():
        if x[i]>threshold:
            count = count + 1
            term.append(i)
    return count,term
#要缩小到一共1200dim
num_disease,disease = filter(Diseases_count,10)
num_tissue,tissue = filter(Tissue_count,10)
num_pathway,pathway = filter(Pathway_count,30)




#####################################################
disease_list = []
# with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/disease_list_30.txt','r') as f:
#     for line in f:
#         disease_list.append(line.strip())
#
# tissue_list = []
# with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/tissue_list_30.txt','r') as f:
#     for line in f:
#         tissue_list.append(line.strip())

# pathway_list = []
# with open('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/pathway_list_30.txt','r') as f:
#     for line in f:
#         pathway_list.append(line.strip())

# disease_features = np.zeros((15133,num_disease))
# tissue_features = np.zeros((15133,num_tissue))
pathway_features = np.zeros((15133,num_pathway))


uniprot = pd.read_pickle( "/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/features.pkl")
#disease
# for id,i in enumerate(uniprot['Entry']):
#     if i in prot_info_new.keys():
#         if len(prot_info_new[i]['Disease-gene associations (DISEASES)'])>0:
#             for j in prot_info_new[i]['Disease-gene associations (DISEASES)']:
#                 if j in disease:
#                     index = disease.index(j)
#                     disease_features[id][index] = 1
# np.save('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/disease_10.npy',disease_features)
# #tissue
# for id,i in enumerate(uniprot['Entry']):
#     if i in prot_info_new.keys():
#         if len(prot_info_new[i]['Tissue expression (TISSUES)'])>0:
#             for j in prot_info_new[i]['Tissue expression (TISSUES)']:
#                 if j in tissue:
#                     index = tissue.index(j)
#                     tissue_features[id][index] = 1
# np.save('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/data_process/tissue_10.npy',tissue_features)
#pathway
for id,i in enumerate(uniprot['Entry']):
    if i in prot_info_new.keys():
        if len(prot_info_new[i]['Reactome Pathways'])>0:
            for j in prot_info_new[i]['Reactome Pathways']:
                if j in pathway:
                    index = pathway.index(j)
                    pathway_features[id][index] = 1
np.save('/home/sgzhang/perl5/GAT-GO/HIF2GO/data/Human/pathway.npy',pathway_features)
print()

