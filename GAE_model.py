import torch.nn as nn
import torch
from utils import create_norm,drop_edge
from torch.nn import functional as F
from base_layers.gat import GAT
from base_layers.gcn import GraphConvolution,GraphConvolutionSparse,dropout_sparse
from base_layers.gin import GIN


######################the implementation of GAE#######################
# def dropout_sparse(x, keep_prob, num_nonzero_elems):#keep_prob=1
#     """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
#     """
#     noise_shape = [num_nonzero_elems]
#     random_tensor = keep_prob
#     random_tensor += torch.rand(noise_shape)
#     return x


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim, dropout=0, act=nn.Sigmoid(), **kwargs):
        super(InnerProductDecoder,self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        x = torch.transpose(inputs,0,1)
        x = torch.matmul(inputs, x)
        x = torch.reshape(x, [1,-1])
        outputs = self.act(x)
        return outputs


# class GraphConvolutionSparse(nn.Module):
#     def __init__(self,input_dim, output_dim,features_nonzero, dropout=0., act=nn.ReLU(), **kwargs):
#         super(GraphConvolutionSparse,self).__init__()
#         w = nn.Parameter(torch.Tensor(input_dim,output_dim))
#         self.weights = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
#         self.dropout = dropout
#         self.act = act
#         self.issparse = True
#         self.features_nonzero = features_nonzero
#     def forward(self,adj,inputs):
#         x = inputs
#         x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
#         x = torch.matmul(x,self.weights)
#         x = torch.matmul(adj,x)
#
#         outputs = self.act(x)
#         return outputs
#
#
# class GraphConvolution(nn.Module):
#     def __init__(self,input_dim, output_dim,dropout=0., act=nn.ReLU(), **kwargs):
#         super(GraphConvolution, self).__init__()
#         w = nn.Parameter(torch.Tensor(input_dim,output_dim))
#         self.weights = nn.init.xavier_uniform_(w,gain=nn.init.calculate_gain('relu'))
#         self.dropout = dropout
#         self.act = act
#     def forward(self,adj,inputs):
#         x = inputs
#         x = torch.matmul(x, self.weights)
#         x = torch.matmul(adj,x)
#         outputs = self.act(x)
#         return outputs




class GAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0, **kwargs):
        super(GAE, self).__init__()

        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.input_dim = num_features
        # self.adj = adj
        # self.inputs = features
        self.device = device
        self.dropout = dropout
        self.features_nonzero = features_nonzero


        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              features_nonzero=self.features_nonzero,
                                              act=nn.ReLU(),
                                              dropout=self.dropout)
        self.hidden2 = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       act=lambda x: x,
                                       dropout=self.dropout)

        self.InnerProductDecoder = InnerProductDecoder(input_dim=self.hidden2_dim,act = lambda x: x)##decoder
    def forward(self,adj,x):
        x1 = self.hidden1(adj, x)

        z = self.hidden2(adj,x1)

        reconstructions = self.InnerProductDecoder(z)

        return z,reconstructions

########################################the implementation of GraphMAE#############
class GCN_encoder(nn.Module):
    def __init__(self,num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0, **kwargs):
        super(GCN_encoder, self).__init__()
        self.hidden1 = GraphConvolutionSparse(input_dim=num_features,
                                              output_dim=hidden1,
                                              features_nonzero=features_nonzero,
                                              act=nn.ReLU(),
                                              dropout=dropout)
        self.hidden2 = GraphConvolution(input_dim=hidden1,
                                        output_dim=hidden2,
                                        act=lambda x: x,
                                        dropout=dropout)
    def forward(self,adj,x):
        x = self.hidden1(adj,x)
        x = self.hidden2(adj,x)
        return x

class GAT_encoder(nn.Module):
    def __init__(self,num_features, num_nodes, features_nonzero, hidden1, hidden2,device,num_layers=2,dropout=0.0, **kwargs):
        super(GAT_encoder, self).__init__()
        self.hidden = GAT(
            in_dim=num_features,
            num_hidden=hidden1,
            out_dim=hidden2,
            num_layers=num_layers,
            nhead=4,
            nhead_out=1,
            concat_out=True,
            activation='relu',
            feat_drop=0.2,
            attn_drop=0.1,
            negative_slope=0.2,
            residual=False,
            norm=create_norm(None),
            encoding=('encoding' == "encoding"),
        )
        self.device = device
    def forward(self,adj,x):
        x = self.hidden(adj,x)
        return x


class GIN_encoder(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, device,num_layers=2, dropout=0.0, **kwargs):
        super(GIN_encoder, self).__init__()
        self.hidden = GIN(
            in_dim=num_features,
            num_hidden=hidden1,
            out_dim=hidden2,
            num_layers=num_layers,
            dropout=dropout,
            activation='relu',
            residual=False,
            norm=create_norm('batchnorm'),
            encoding=('encoding' == "encoding"),
        )
    def forward(self, adj, x):
        x = self.hidden(adj,x)
        return x

class MLP_decoder(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0, **kwargs):
        super(MLP_decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, num_features))
    def forward(self,x):
        x = self.decoder(x)
        return x

class GCN_decoder(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0, **kwargs):
        super(GCN_decoder, self).__init__()
        self.hidden = GraphConvolution(input_dim=hidden2,
                                        output_dim=num_features,
                                        act=lambda x: x,
                                        dropout=dropout)
    def forward(self,adj,x):
        x = self.hidden(adj,x)
        return x
class GraphMAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0,mask_rate=None,enc_type=None,dec_type=None, **kwargs):
        super(GraphMAE, self).__init__()
        self.InnerProductDecoder = InnerProductDecoder(input_dim=hidden2, act=lambda x: x)
        self._replace_rate = 0.05#0.05
        self.enc_mask_token = nn.Parameter(torch.zeros(1, num_features))
        self.mask_rate = mask_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.enc_type = enc_type
        self.dec_type = dec_type
        #encoder-decoder
        if enc_type == 'GCN':
            self.encoder = GCN_encoder(num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0)
        elif enc_type == 'GAT':
            self.encoder = GAT_encoder(num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0)
        elif enc_type == 'GIN':
            self.encoder = GIN_encoder(num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0)
        else:
            raise Exception('The encoder model ' + enc_type + ' are not supported!')
        self.encoder_to_decoder = nn.Linear(hidden2, hidden2, bias=False)
        if dec_type == 'MLP':
            self.decoder = MLP_decoder(num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0)
        elif dec_type == 'GCN':
            self.decoder = GCN_decoder(num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0)
        elif dec_type == 'GAT':
            self.decoder = GAT_encoder(hidden2, num_nodes, features_nonzero, hidden1, num_features,device, dropout=0.0,num_layers=1)
        elif dec_type == 'GIN':
            self.decoder = GIN_encoder(hidden2, num_nodes, features_nonzero, hidden1, num_features,device, dropout=0.0,num_layers=1)
        else:
            raise Exception('The decoder model ' + enc_type + ' are not supported!')
        ############################

        self.n_samples = num_nodes

    def encoding_mask_noise(self, adj, x, mask_rate):
        #对原始特征进行mask
        num_nodes = self.n_samples#全部结点数
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = adj.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)


    def forward(self, adj, x):

        adj, mask_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self.mask_rate)#mask_feat

        z_with_mask = self.encoder(adj,mask_x)

        rep = self.encoder_to_decoder(z_with_mask)

        if self.dec_type == 'MLP':
            recon = self.decoder(rep)
        else:
            rep[mask_nodes] = 0
            recon = self.decoder(adj,rep)

        x_init = x[mask_nodes]

        x_rec = recon[mask_nodes]

        z = self.encoder(adj,x)

        adj_rec = self.InnerProductDecoder(z)
        return x_init,x_rec,z,z_with_mask,adj_rec##############rep or z
####################################the implementation of ARGMA#################################################

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


EPS = 1e-15
class ARGMA(nn.Module):
    def __init__(self,encoder,decoder,discriminitor,num_nodes,num_features,mask_rate):
        super(ARGMA, self).__init__()
        self.n_samples = num_nodes
        self._replace_rate = 0.05
        self.enc_mask_token = nn.Parameter(torch.zeros(1, num_features))
        self.mask_rate = mask_rate
        self._mask_token_rate = 1 - self._replace_rate

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminitor
        self.encoder_to_decoder = nn.Linear(400,400, bias=False)
    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss


    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encoding_mask_noise(self, adj, x, mask_rate):
        #对原始特征进行mask
        num_nodes = self.n_samples#全部结点数
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = adj.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)


