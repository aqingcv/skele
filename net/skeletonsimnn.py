import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class SkeletonSimnn(nn.Module):

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, n_neighbors=1,
                 mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.k = n_neighbors
        if not self.pretrain:
            self.encoder_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.encoder_joint = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            
            if mlp:  # hack: brute-force replacement
                dim_hidden = self.encoder_joint.fc.weight.shape[1]
                self.encoder_joint.fc = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  self.encoder_joint.fc,
                                                  nn.BatchNorm1d(feature_dim))
                self.predictor_joint = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                                        nn.BatchNorm1d(feature_dim),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(feature_dim, feature_dim)) 

                self.encoder_motion.fc = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(dim_hidden, dim_hidden),
                                                  nn.BatchNorm1d(dim_hidden),
                                                  nn.ReLU(inplace=True),
                                                  self.encoder_motion.fc,
                                                  nn.BatchNorm1d(feature_dim))

                self.predictor_motion = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                                        nn.BatchNorm1d(feature_dim),
                                                        nn.ReLU(inplace=True),
                                                        nn.Linear(feature_dim, feature_dim)) 

                # self.predictor = nn.Sequential(nn.Linear(dim_out, dim_out//2),
                #                         nn.BatchNorm1d(dim_out//2),
                #                         nn.ReLU(inplace=True),
                #                         nn.Linear(dim_out//2, dim_out)) 

    # def _compute_distance(self, x, y):
    #     x = F.normalize(x, dim = 1)
    #     y = F.normalize(y, dim = 1)
    #     dist = torch.einsum("nd,md->nm",x, y)
    #     return dist
    def _compute_distance(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)

        dist = 2 - 2 * torch.sum(x.view(x.shape[0], 1, x.shape[1]) *
                                    y.view(1, y.shape[0], y.shape[1]), -1)
        return dist

    def _knn(self, x, y):
        # compute distance
        dist = self._compute_distance(x, y)

        # compute k nearest neighbors
        _, indices = torch.topk(dist, k=self.k, largest=False)

        # randomly select one of the neighbors
        selection_mask = torch.randint(self.k, size=(indices.size(0),))
        mined_views_ids = indices[torch.arange(indices.size(0)).to(selection_mask), selection_mask]
        return mined_views_ids

    def mine_views(self, x, y_pool):
        r"""Finds, for each element in batch :obj:`y`, its nearest neighbors in :obj:`y_pool`, randomly selects one
            of them and returns the corresponding index.

        Args:
            y (torch.Tensor): batch of representation vectors.
            y_pool (torch.Tensor): pool of candidate representation vectors.

        Returns:
            torch.Tensor: Indices of mined views in :obj:`y_pool`.
        """


        mined_views_ids = self._knn(y, y_pool)
        return mined_views_ids
                   
    def forward(self, im_1, im_2 = None):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        """

        im_1_motion = torch.zeros_like(im_1)
        im_1_motion[:, :, :-1, :, :] = im_1[:, :, 1:, :, :] - im_1[:, :, :-1, :, :]

        if not self.pretrain:
            return (self.encoder_joint(im_1) + self.encoder_motion(im_1_motion)) / 2.

        
        im_2_motion = torch.zeros_like(im_2)
        im_2_motion[:, :, :-1, :, :] = im_2[:, :, 1:, :, :] - im_2[:, :, :-1, :, :]

        # compute online features
        z1 = self.encoder_joint(im_1)
        z2 = self.encoder_joint(im_2)
        z3 = self.encoder_motion(im_1_motion)
        z4 = self.encoder_motion(im_2_motion)

        ids1 = self.mine_views(z1,z2)
        ids2 = self.mine_views(z2,z1)
        ids3 = self.mine_views(z3,z4)
        ids4 = self.mine_views(z4,z3)

        n1 = z1[ids1].contiguous()
        n2 = z2[ids2].contiguous()
        n3 = z3[ids3].contiguous()
        n4 = z4[ids4].contiguous()

        p1 = self.predictor_joint(z1)
        p2 = self.predictor_joint(z2)
        p3 = self.predictor_motion(z3)
        p4 = self.predictor_motion(z4)  
       
        return p1, p2, p3, p4, z1.detach(), z2.detach(), z3.detach(), z4.detach(), n1.detach(), n2.detach(), n3.detach(), n4.detach()
        