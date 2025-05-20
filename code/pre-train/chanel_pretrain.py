import time

import torch
import collections
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model.MaskedAutoEncoderModel import MaskedViTAutoEncoder
import os
import torch.optim as optim
import torch.nn.functional as F


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = 'cuda'
save_path="your_save_path"
data_path="your_data_path"
data=np.load(data_path+"\\SpatialInput.npy",allow_pickle=True)
data=torch.FloatTensor(data).to(device)
dataset=TensorDataset(data)
dataloader=DataLoader(dataset,batch_size=280,shuffle=True)

mae=MaskedViTAutoEncoder(image_size=9,patch_size=1,in_channels=1,embed_dim=32,num_heads=4,num_layers=12,mask_ratio=0.6).to(device)
optimizer=optim.Adam(mae.parameters(),lr=1e-3)
epoch=5000
best_loss = float('inf')

for epoch_idx in range(epoch):
    loss_item = []
    start_time = time.time()

    with tqdm(total=len(dataloader),
              desc=f"Epoch {epoch_idx + 1}/{epoch}",
              unit="batch") as pbar:

        for x in dataloader:
            x = x[0]
            optimizer.zero_grad()
            loss = mae.train_chanel_loss(x)
            loss.backward()
            optimizer.step()
            loss_item.append(loss.item())
            avg_loss = sum(loss_item) / len(loss_item)
            pbar.set_postfix({
                'Batch Loss': f"{loss.item():.4f}",
                'Running Loss': f"{avg_loss:.4f}"
            })
            pbar.update(1)

    avg_loss = sum(loss_item) / len(loss_item)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(mae.state_dict(), save_path + "\\BestMaskedViTAutoEncoder_chanel_93_4_514.pth")

    end_time = time.time()
    train_time = end_time - start_time

torch.save(mae.state_dict(),save_path+"\\MaskedViTAutoEncoder_chanel_93_4_514.pth")

