import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


# def mine_views(output , y_pool):

#     output_normed = torch.nn.functional.normalize(output, dim=1)
#     bank_normed = torch.nn.functional.normalize(y_pool, dim=1)
#     similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed)
#     _, indices = torch.topk(similarity_matrix, 2)
#     # randomly select one of the neighbors
#     selection_mask = torch.randint(2, size=(indices.size(0),))
#     mined_views_ids = indices[torch.arange(indices.size(0)).to(selection_mask), selection_mask]

#     return y_pool[mined_views_ids]

class SkeletonSim(nn.Module):

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, 
                 mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_hidden= self.encoder.fc.weight.shape[1]
                self.encoder.fc = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  self.encoder.fc,
                                                  nn.BatchNorm1d(feature_dim))

                self.predictor = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                                        nn.BatchNorm1d(feature_dim),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(feature_dim, feature_dim))                                    


        # index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        # nearest_neighbours = torch.index_select(y_pool, dim=0, index=index_nearest_neighbours)

        # return nearest_neighbours

    def forward(self, im_1, im_2 = None):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        """
        # im_1_motion = torch.zeros_like(im_1)
        # im_1_motion[:, :, :-1, :, :] = im_1[:, :, 1:, :, :] - im_1[:, :, :-1, :, :]

        if not self.pretrain:
            return self.encoder(im_1)

        # compute online features
        z1 = self.encoder(im_1)
        z2 = self.encoder(im_2)

        # n1 = mine_views(z1,z1)
        # n2 = mine_views(z2,z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)       

        return p1, p2, z1.detach(), z2.detach()
        