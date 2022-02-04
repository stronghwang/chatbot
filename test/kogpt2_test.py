import torch

from transformers import PreTrainedTokenizerFast

ctx = 'cuda:0'
device = torch.device(ctx)
model = torch.load('../output/model/chatbot.pt')
model.to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',)
q_token = "<usr>"
a_token = "<sys>"
sent_token = '<unused1>'

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(
                tokenizer.encode(q_token + q + sent_token + a_token + a)).unsqueeze(dim=0)
            if ctx == 'cuda:0':
                input_ids = input_ids.to(ctx)
            pred = model(input_ids)
            pred = pred.logits
            if ctx == 'cuda:0':
                pred = pred.cpu()
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == tokenizer.eos_token:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))