import csv
import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def get_first_n_words(string, n):
    words = string.split()
    result = words[:n]
    return ' '.join(result)

class Lyrics(Dataset):
    def __init__(self, path: str, tokenizer, max_length=1022):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lyrics = []
        self.load_csv_file(path)
        self.lyrics = pad_sequence(self.lyrics)
        #print("self.data: ", self.data )
        #self.data_encoded = tokenizer.encode(self.data, truncation=True, padding=True)
        #print("self.lyrics: ", self.lyrics)
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, idx):
        #print("Im inside getitem")
        #print("self.input_ids[idx]: ", self.input_ids[idx])
        #print("self.attention_mask[idx]: ", self.attention_mask[idx])
        return self.lyrics[idx]

    def load_csv_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the first row as the header
            count = 0
            for row in reader:
                print("&"*50)
                print("row[-1]: ", row[-1])
                print("self.get_first_n_words(row[-1], self.max_length): ", get_first_n_words(row[-1], self.max_length))
                print("&"*50)
                #print("row[-1][:max_length]: ", row[-1][:self.max_length])
                if(count > 0):
                    break
                self.lyrics.append(torch.tensor(
                    self.tokenizer.encode('<startofstring> ' + get_first_n_words(row[-1], self.max_length) + ' <endofstring>')
                ))
                count = count + 1
    
