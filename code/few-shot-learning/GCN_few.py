import random

import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from model.Predict import Predict,get_quantile_values

class GCN_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN_Model, self).__init__()
        self.gcn1=pyg_nn.GCNConv(input_dim, 128)
        self.gcn2=pyg_nn.GCNConv(128, 32)
        self.gcn3=pyg_nn.GCNConv(32, 16)
        self.linear1=nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = 'cuda'

data_path="your_data_path"
save_path="your_save_path"

data=np.load(data_path+"\\GraphInput.npy",allow_pickle=True)
adj=np.load(data_path+"\\adj.npy",allow_pickle=True)
label=np.load(data_path+"\\label.npy",allow_pickle=True)[:,0]
positive_index=np.where(label==1)[0]
negative_index=np.where(label==0)[0]
data=torch.FloatTensor(data).to(device)
label=torch.LongTensor(label).to(device)
adj=torch.LongTensor(adj).to(device)

hidden_dims=39
classifier=GCN_Model(hidden_dims,2).to(device)
classifier.train()
optimizer=optim.Adam(classifier.parameters(),lr=1e-3)
epoch=300

hidden_feature=data
for i in range(epoch):
    optimizer.zero_grad()
    out=classifier(hidden_feature,adj)
    loss=F.cross_entropy(out[positive_index],label[positive_index])+F.cross_entropy(out[negative_index],label[negative_index])
    loss.backward()
    optimizer.step()

    print("epoch:{},train loss:{}".format(i,loss.item()))

data=np.load(data_path+"\\SpectrumInput.npy",allow_pickle=True)
XY=np.load(data_path+"\\XY.npy",allow_pickle=True).T
data=torch.FloatTensor(data).to(device)
with torch.no_grad():
    hidden_feature=data
    out=classifier(hidden_feature,adj)
    out=F.softmax(out,dim=1)
out = out.to("cpu").detach().numpy()
out = (out[:,1]-out[:,1].min())/(out[:,1].max()-out[:,1].min())
Predict(out,XY,save_path+"\\GCN.tif")
out,out2 = get_quantile_values(out,5)
Predict(out,XY,save_path+"\\GCN_reclassify.tif")
Predict(out2,XY,save_path+"\\GCN_class.tif")