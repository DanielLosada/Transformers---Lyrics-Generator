import pandas as pd
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from lyrics_dataset import SongLyrics
import torch.nn.functional as F
import csv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    acc_steps = 100
    #device=torch.device("cuda")
    #model = model.cuda()
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model

if __name__ == "__main__":
    ### Prepare data
    lyrics = pd.read_csv('dataset/lyrics-data.csv')
    lyrics = lyrics[lyrics['language']=='en']

    # Only keep popular artists, with genre Rock/Pop and popularity high enough
    artists = pd.read_csv('dataset/artists-data.csv')
    artists = artists[(artists['Genres'].isin(['Rock'])) & (artists['Popularity']>5)]
    df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
    df = df.drop(columns=['ALink','SLink','language','Link'])

    # Drop the songs with lyrics too long (after more than 1024 tokens, does not work)
    df = df[df['Lyric'].apply(lambda x: len(x.split(' ')) < 350)]

    # Create a very small test set to compare generated text with the reality
    test_set = df.sample(n = 200)
    df = df.loc[~df.index.isin(test_set.index)]

    # Reset the indexes
    test_set = test_set.reset_index()
    df = df.reset_index()

    # For the test set only, keep last 20 words in a new column, then remove them from original column
    test_set['True_end_lyrics'] = test_set['Lyric'].str.split().str[-20:].apply(' '.join)
    test_set['Lyric'] = test_set['Lyric'].str.split().str[:-20].apply(' '.join)

    # Create dataset object
    dataset = SongLyrics(df['Lyric'], df, truncate=True, gpt2_type="gpt2")

    # Get the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Train the model
    model = train(dataset, model, tokenizer)
