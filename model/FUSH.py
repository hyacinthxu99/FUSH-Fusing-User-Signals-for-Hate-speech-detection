import numpy as np
import torch_geometric
from sklearn import metrics
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from HGT import HGTEncoder


class BilinearFusion(nn.Module):
    def __init__(self, graph_dim, text_dim, out_dim):
        super(BilinearFusion, self).__init__()
        self.bilinear = nn.Bilinear(graph_dim, text_dim, out_dim)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()

    def forward(self, graph_feat, text_feat):
        """
        graph_feat: Tensor [B, Dg]
        text_feat:  Tensor [B, Dt]
        return:	 Tensor [B, Dout]
        """
        fused = self.bilinear(graph_feat, text_feat)
        return self.act(self.dropout(fused))


class FUSH(nn.Module):
    def __init__(self, model_name, num_labels, node_feat_dim, metadata, edge_weight_dict, text_feat_dim=1024,
                 gcn_hidden=128, fusion_hidden=64):
        super(FUSH, self).__init__()

        self.device = torch.device("cuda")

        # 加载预训练的 BERT 模型（不带分类头）
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.gcn_encoder = HGTEncoder(
            in_dim=node_feat_dim,
            hidden_dim=64,
            out_dim=gcn_hidden,
            metadata=metadata,
            edge_weight_dict=edge_weight_dict,
            num_heads=4,
            num_layers=2
        )

        self.bilinear = BilinearFusion(text_feat_dim, gcn_hidden * 2, fusion_hidden)

        self.classifier = nn.Linear(fusion_hidden, num_labels)

        self.dropout = nn.Dropout(0.2)


    def forward(self, x_dict, edge_index_dict, edge_weight_dict, input_ids, replied_node_ids, comment_node_ids,
                attention_mask=None, token_type_ids=None):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs.last_hidden_state[:, 0]

        gcn_embeds = self.gcn_encoder(x_dict, edge_index_dict)  # [num_users, gcn_hidden]

        replied_tensor = gcn_embeds[replied_node_ids]  # [batch_size, gcn_hidden]
        comment_tensor = gcn_embeds[comment_node_ids]  # [batch_size, gcn_hidden]

        user_tensor = torch.cat((replied_tensor, comment_tensor), dim=1)

        embedding = self.bilinear(pooled_output, user_tensor)
        logits = self.classifier(embedding)

        return logits, user_tensor





