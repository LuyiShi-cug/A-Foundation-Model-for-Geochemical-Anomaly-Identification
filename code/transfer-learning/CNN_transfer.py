import random
import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from model.Predict import Predict,get_quantile_values
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
save_path="your_save_path"

data=np.load(source_path+"\\SpatialInput.npy",allow_pickle=True)
label=np.load(source_path+"\\label.npy",allow_pickle=True)[:,0]

positive_index=np.where(label==1)[0]
negative_index=np.where(label==0)[0]

data=torch.FloatTensor(data).to(device)
label=torch.LongTensor(label).to(device)



# hidden_dims=25*192
hidden_dims=39
classifier=nn.Sequential(
                        nn.Conv2d(hidden_dims,128,5,1),nn.ReLU(),
    nn.Conv2d(128,64,3,1),nn.ReLU(),
    nn.Conv2d(64,32,3,1),nn.ReLU(),nn.Flatten(),
                         nn.Linear(32,8),
                         nn.ReLU(),
                         nn.Linear(8,2)).to(device)

classifier.train()
optimizer=optim.Adam(classifier.parameters(),lr=1e-3)
epoch=100

hidden_feature=data
for i in range(epoch):
    optimizer.zero_grad()
    out=classifier(hidden_feature)
    loss=F.cross_entropy(out[positive_index],label[positive_index])+F.cross_entropy(out[negative_index],label[negative_index])
    loss.backward()
    optimizer.step()

    print("epoch:{},train loss:{}".format(i,loss.item()))


data=np.load(target_path+"\\SpatialInput.npy",allow_pickle=True)
XY=np.load(target_path+"\\XY.npy",allow_pickle=True).T
data=torch.FloatTensor(data).to(device)
with torch.no_grad():
    hidden_feature=data
    out=classifier(hidden_feature)
    out=F.softmax(out,dim=1)
out = out.to("cpu").detach().numpy()

out = (out[:,1]-out[:,1].min())/(out[:,1].max()-out[:,1].min())
Predict(out,XY,save_path+"\\CNN.tif")
out,out2 = get_quantile_values(out,5)
Predict(out,XY,save_path+"\\CNN_reclassify.tif")
Predict(out2,XY,save_path+"\\CNN_class.tif")