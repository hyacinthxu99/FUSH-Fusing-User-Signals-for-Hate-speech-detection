from torch_geometric.nn import HGTConv
import torch
import torch.nn as nn

class HGTEncoder(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, metadata, edge_weight_dict,num_heads=4, num_layers=2):
		"""
		Heterogeneous Graph Transformer Encoder

		参数:
		- in_dim: 输入特征维度
		- hidden_dim: 中间层维度
		- out_dim: 最终输出特征维度
		- metadata: (node_types, edge_types)
		- num_heads: 多头注意力头数
		- num_layers: 层数
		"""
		super().__init__()
		self.layers = nn.ModuleList()
		self.edge_weight_dict = edge_weight_dict

		for i in range(num_layers):
			self.layers.append(HGTConv(
				in_channels=in_dim if i == 0 else hidden_dim,
				out_channels=hidden_dim if i < num_layers - 1 else out_dim,
				metadata=metadata,
				heads=num_heads
			))

	def forward(self, x_dict, edge_index_dict,edge_weight_dict = None):
		"""
		输入:
		- x_dict: {'user': node_features}
		- edge_index_dict: {('user', 'like', 'user'): edge_index_like, ...}
		返回:
		- x_dict: {'user': user_embedding}
		"""
		for conv in self.layers:
			if edge_weight_dict == None:
				x_dict = conv(x_dict, edge_index_dict)
			else:
				x_dict = conv(x_dict, edge_index_dict)
		return x_dict["user"]