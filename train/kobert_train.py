import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataloader.dataloader import BERTDataset
from model.kobert_model import BERTClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

dataset_train = []
file = open('../input/test.txt', 'r', encoding='utf-8')
while True:
    line = file.readline()
    if not line:
        break
    dataset_train.append([line[:-3], line[-2]])
file.close()
print(len(dataset_train))


max_len = 64
batch_size = 8
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
learning_rate = 5e-5

model = BERTClassifier()
model.to(device)
train_dataset = BERTDataset(dataset_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


for epoch in range(num_epochs):
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        optimizer.zero_grad()
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, '../output/model/cp_badword.pt')

model.eval()
torch.save(model, '../output/model/badword.pt')
