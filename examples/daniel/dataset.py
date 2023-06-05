import csv
from torch.utils.data import Dataset

class Lyrics(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = self.load_csv_file(path)
        #print("self.data: ", self.data )
        self.data_encoded = tokenizer(self.data, truncation=True, padding=True)
        self.input_ids = self.data_encoded['input_ids']
        self.attention_mask = self.data_encoded['attention_mask']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print("Im inside getitem")
        #print("self.input_ids[idx]: ", self.input_ids[idx])
        #print("self.attention_mask[idx]: ", self.attention_mask[idx])
        return (self.input_ids[idx], self.attention_mask[idx])

    def load_csv_file(self, file_path):
        lyrics = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the first row as the header
            for row in reader:
                lyrics.append('<startofstring> ' + row[-1] + ' <endofstring>')     
        return lyrics

