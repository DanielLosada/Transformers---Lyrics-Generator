import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from dataset import Lyrics
from torch.optim import Adam
from torch.utils.data import DataLoader

def train(lyricsLoader, model, optim):
    model.train()
    epochs = 10

    for i in tqdm.tqdm(range(epochs)):
        for X, a in lyricsLoader:
            print("X: ", type(X))
            print("len X: ", len(X))
            #print("a: ", a)
            #X = torch.tensor(X).to(device)
            #a = torch.tensor(a).to(device)
            optim.zero_grad()

            loss = model(X, attention_mask=a)
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
            
def infer(inp):
    inp = '<startofstring> ' + inp + ' <endofstring>'
    inp = tokenizer(inp)
    output = model.generate(**inp)
    output = tokenizer.decode(output[0])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.add_special_tokens({
    "bos_token": '<startofstring>',
    "eos_token": '<endofstring>',
    'pad_token': '[PAD]'
})

model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
#model.to(device)

#print("tokenizer.eos_token_id: ", tokenizer.eos_token_id)
#print(tokenizer.decode(tokenizer.eos_token_id))
#print("tokenizer.bos_token_id: ", tokenizer.bos_token_id)
#print(tokenizer.decode(tokenizer.bos_token_id))


lyrics = Lyrics('./archive/AAP Rocky.csv', tokenizer)
#print("lyrics: ", lyrics)
lyricsLoader = DataLoader(lyrics, batch_size=64)
#print("lyricsLoader: ", lyricsLoader)
optim = Adam(model.parameters())

train(lyricsLoader, model, optim)
#sequence = f"Hello "

#input = tokenizer.encode(sequence, return_tensors="pt")
#generated = model.generate(input, max_length=100, do_sample=True, no_repeat_ngram_size=2)
#print(tokenizer.decode(generated[0], skip_special_tokens=True))

