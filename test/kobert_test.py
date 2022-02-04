import torch
import numpy as np
import gluonnlp as nlp

from kobert_tokenizer import KoBERTTokenizer
from model.kobert_model import BERTClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
model = torch.load('../output/model/badword.pt')
model.to(device)


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100


def testModel(model, sentence):
    cate = ['비욕설', '욕설']
    transform = nlp.data.BERTSentenceTransform(tokenizer.tokenize, 64, vocab, True, False)
    tokenized = transform([sentence])
    result = model(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
    idx = result.argmax().cpu().item()
    print("문장에는:", cate[idx])
    print("신뢰도는:", "{:.2f}%".format(softmax(result, idx)))


while True:
    s = input('input: ')
    if s == 'quit':
        break
    testModel(model, s)
