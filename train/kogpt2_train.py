import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

from dataloader.dataloader import GPT2Dataset
from model.kogpt2_model import GPT2Chat


# ctx = "cuda:0"
ctx = "cpu"
print(ctx)
device = torch.device(ctx)


keti_file = open('../input/KETI_office_dataset_for_autoregressive.txt', 'r', encoding='utf-8')
Q = []
A = []
while True:
    line = keti_file.readline()
    if not line:
        break
    datas = line.split("    ")
    Q.append(datas[0])
    A.append(datas[1][:-1])
keti_Data = pd.DataFrame({'Q': Q, 'A': A})
keti_Data.info()


epochs = 40
batch_size = 8
Sneg = -1e18
learning_rate = 3e-5

model = GPT2Chat()
model.to(device)
train_dataset = GPT2Dataset(keti_Data)

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_batch,)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(reduction="none")


model.train()
for epoch in range(epochs):
    for batch_idx, (token_ids, mask, label) in enumerate(tqdm(train_dataloader)):
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        out = model(token_ids).logits
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        # 학습 끝
        optimizer.step()
    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, '../output/model/cp_chatbot.pt')
    print('epoch:', epoch, 'loss:', avg_loss)

model.eval()
torch.save(model, '../output/model/chatbot.pt')