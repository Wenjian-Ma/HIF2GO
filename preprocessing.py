from torch_geometric.data import InMemoryDataset
import torch

class PFPDataset(InMemoryDataset):
    def __init__(self, dir=None, train_data_X=None,train_data_Y=None,lm=None,transform=None,pre_transform=None):

        super(PFPDataset, self).__init__( transform, pre_transform)
        self.dir=dir
        self.X_data_list = train_data_X
        self.Y_data_list = train_data_Y
        self.lm = lm
    def __len__(self):
        return int(self.X_data_list.shape[0])

    def __getitem__(self, idx):
        embedding = self.X_data_list[idx]
        lm = self.lm[idx]
        label = self.Y_data_list[idx]

        embedding = torch.Tensor([embedding])

        lm = torch.Tensor([lm])

        label = torch.Tensor([label])

        return embedding,label,lm

def collate(data_list):
    embedding = [data[0] for data in data_list]

    label = [data[1] for data in data_list]

    lm = [data[2] for data in data_list]

    embedding_list = torch.stack(embedding).squeeze(dim=1)

    label_list = torch.stack(label).squeeze(dim=1)

    lm_list = torch.stack(lm).squeeze(dim=1)

    return embedding_list,label_list,lm_list
