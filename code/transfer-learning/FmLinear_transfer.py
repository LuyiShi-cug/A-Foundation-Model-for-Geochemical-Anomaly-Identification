import random

import torch
import numpy as np

from model.MaskedAutoEncoderModel import MaskedViTAutoEncoder
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model.Predict import Predict, get_quantile_values

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
device = 'cuda'

source_path="your_source_path"
target_path="your_target_path"
model_path="your_model_path"
save_path="your_save_path"

data=np.load(source_path+"\\SpatialInput.npy",allow_pickle=True)
label=np.load(source_path+"\\label.npy",allow_pickle=True)[:,0]

data=torch.FloatTensor(data[label>-1]).to(device)
label=torch.LongTensor(label[label>-1]).to(device)


mae=MaskedViTAutoEncoder(image_size=9,patch_size=1,in_channels=1,embed_dim=32,num_heads=4,num_layers=12,mask_ratio=0.6).to(device)
mae.load_state_dict(torch.load(model_path))

classifier=nn.Sequential(
                         nn.Linear(16*32,128),
                         nn.ReLU(),
                         nn.Linear(128,32),
                         nn.ReLU(),
                         nn.Linear(32,8),
                         nn.ReLU(),
                         nn.Linear(8,2)).to(device)

mae.eval()
classifier.train()
optimizer=optim.Adam(classifier.parameters(),lr=1e-3)
epoch=500
with torch.no_grad():
    hidden_feature=mae.calculate_latent(data)
    hidden_feature=hidden_feature.mean(2)
    hidden_feature=hidden_feature.reshape(hidden_feature.shape[0],-1)
    hidden_feature=hidden_feature.detach()
for i in range(epoch):
    optimizer.zero_grad()
    out=classifier(hidden_feature)
    loss=F.cross_entropy(out,label)
    loss.backward()
    optimizer.step()

    print("epoch:{},train loss:{}".format(i,loss.item()))

data=np.load(target_path+"\\SpatialInput.npy",allow_pickle=True)
XY=np.load(target_path+"\\XY.npy",allow_pickle=True).T
data=torch.FloatTensor(data).to(device)
with torch.no_grad():
    hidden_feature=mae.calculate_latent(data)
    hidden_feature=hidden_feature.mean(2)
    hidden_feature=hidden_feature.reshape(hidden_feature.shape[0],-1)
    hidden_feature=hidden_feature.detach()

    out=classifier(hidden_feature)
    out=F.softmax(out,dim=1)
out = out.to("cpu").detach().numpy()
out = (out[:,1]-out[:,1].min())/(out[:,1].max()-out[:,1].min())
Predict(out,XY,save_path+"\\FMLinear.tif")
out,out2 = get_quantile_values(out,5)
Predict(out,XY,save_path+"\\FMLinear_reclassify.tif")
Predict(out2,XY,save_path+"\\FMLinear_class.tif")