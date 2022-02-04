import torch
import gluonnlp as nlp

from model.kobert_model import BERTClassifier
from kobert_tokenizer import KoBERTTokenizer
from transformers import PreTrainedTokenizerFast

ctx = 'cuda:0'
device = torch.device(ctx)

badword_model = torch.load('./output/model/badword.pt')
badword_model.to(device)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tok = tokenizer.tokenize
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

chat_model = torch.load('./output/model/chatbot.pt')
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            transform = nlp.data.BERTSentenceTransform(tok, 64, vocab, True, False)
            tokenized = transform([q])
            result = badword_model(torch.tensor(
                [tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
            idx = result.argmax().cpu().item()
            if idx == 1:
                a = "비속어는 안돼요!"
                break

            input_ids = torch.LongTensor(
                koGPT2_TOKENIZER.encode("<usr>" + q + '<unused1>' + "<sys>" + a)).unsqueeze(dim=0)
            input_ids = input_ids.to(ctx)
            pred = chat_model(input_ids)
            pred = pred.logits
            pred = pred.cpu()
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == '</s>':
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))
