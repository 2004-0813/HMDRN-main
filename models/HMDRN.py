import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import DConv_4, DResNet
from .backbones.MTFEM import MTFEM
from .backbones.CLARM import CLARM


class HMDRN(nn.Module):
    def __init__(self, way=None, shots=None, resnet=False):
        super().__init__()

        self.resolution1 = 5 * 5
        self.resolution2 = 10 * 10

        if resnet:
            self.num_channel1 = 640
            self.num_channel2 = 320
            self.feature_extractor = DResNet.resnet12()
            self.dim = self.num_channel1 * 5 * 5

        else:
            self.num_channel1 = 64
            self.num_channel2 = 64
            self.feature_extractor = DConv_4.BackBone(self.num_channel1)
            self.dim = self.num_channel1 * 5 * 5

        self.mtfem1 = MTFEM(
                sequence_length=self.resolution1,
                embedding_dim=self.num_channel1,
                num_layers=3,
                num_heads=2,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.mtfem2 = MTFEM(
                sequence_length=self.resolution2,
                embedding_dim=self.num_channel2,
                num_layers=3,
                num_heads=2,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.clarm1 = CLARM(
            hidden_size=self.num_channel1,
            inner_size=self.num_channel1,
            num_patch=self.resolution1,
            drop_prob=0.1
        )

        self.clarm2 = CLARM(
            hidden_size=self.num_channel2,
            inner_size=self.num_channel2,
            num_patch=self.resolution2,
            drop_prob=0.1
        )

        self.shots = shots
        self.way = way
        self.resnet = resnet
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

    def get_feature_vector(self, inp, way, shot):
        batch_size = inp.size(0)

        feature_map2, feature_map1 = self.feature_extractor(inp)
        split_idx = way * shot

        support_features1 = feature_map1[:split_idx]
        query_features1 = feature_map1[split_idx:]

        support_features1 = self.mtfem1(support_features1, is_query=False)
        query_features1 = self.mtfem1(query_features1, is_query=True)

        support_features1 = support_features1.transpose(1, 2).view(
            support_features1.size(0), self.num_channel1, 5, 5
        )
        query_features1 = query_features1.transpose(1, 2).view(
            query_features1.size(0), self.num_channel1, 5, 5
        )

        support_features2 = feature_map2[:split_idx]
        query_features2 = feature_map2[split_idx:]

        support_features2 = self.mtfem2(support_features2, is_query=False)
        query_features2 = self.mtfem2(query_features2, is_query=True)

        support_features2 = support_features2.transpose(1, 2).view(
            support_features2.size(0), self.num_channel2, 10, 10
        )
        query_features2 = query_features2.transpose(1, 2).view(
            query_features2.size(0), self.num_channel2, 10, 10
        )

        return support_features1, query_features1, support_features2, query_features2

    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        support_features1, query_features1, support_features2, query_features2 = self.get_feature_vector(inp, way, shot)

        support_features1 = support_features1.view(
            way, shot, *support_features1.size()[1:]
        ).permute(0, 2, 1, 3, 4).contiguous()

        support_features2 = support_features2.view(
            way, shot, *support_features2.size()[1:]
        ).permute(0, 2, 1, 3, 4).contiguous()

        similarity1 = self.clarm1(
            support_features1,
            query_features1
        )

        similarity2 = self.clarm2(
            support_features2,
            query_features2
        )

        similarity = self.w1 * similarity1 + self.w2 * similarity2

        return similarity

    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist = self.get_neg_l2_dist(
            inp=inp,
            way=way,
            shot=shot,
            query_shot=query_shot
        )
        _, max_index = torch.max(neg_l2_dist, 1)
        return max_index

    def forward(self, inp):
        logits = self.get_neg_l2_dist(
            inp=inp,
            way=self.way,
            shot=self.shots[0],
            query_shot=self.shots[1]
        )

        logits = logits / self.dim * self.scale

        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction