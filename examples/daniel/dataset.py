import re
import os
import zipfile

from datasets import load_dataset, concatenate_datasets

files_to_delete = ["(Scriptonite)", "BTS", "Damso", "Genius English Translations", "Genius Romanizations", "JuL", "Nekfeu", "Oxxxymiron"]
files_to_multiartist = ["50 Cent", "Imagine Dragons", "Justin Bieber", "Taylor Swift", "Queen", "Lil Peep", "Arctic Monkeys", "The Notorious B.I.G.", "Radiohead", "Mac Miller"]

# Preprocessing function
def preprocess_lyrics(data):
    # Remove the first line
    data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
    
    # Remove text between square brackets
    data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
    data['lyrics'] = data['lyrics'].strip()
    return data

def preprocess_lyrics_multiple_artists(data):
    data = preprocess_lyrics(data)
    data['lyrics'] =  data['artist'] + ": " + data['lyrics']
    return data

class LyricsDataset():
    
    def __init__(self, config, artist = ""):
        self.config = config
        self.artist = artist

        if not os.path.exists('data') or not os.listdir('data'):
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.config["dataset_zip"]), 'r') as zip_ref:
                zip_ref.extractall(self.config["dataset_path"])
                print("Successfully extracted the contents of the zip file.")
            
        for file_name in files_to_delete:
            file_path = os.path.join(self.config["dataset_path"], f"{file_name}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        else:
            print("The 'data' folder is not empty. Skipping extraction.")
    
    def load_dataset_single_artist(self):
        csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.artist + ".csv")
        csvFile = load_dataset("csv", data_files=csv_path, split="train")
        csvFile = csvFile.map(preprocess_lyrics)
        dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)
        return dataset
    
    def load_dataset_multiple_artists(self):
        folder_path = os.path.join(self.config["base_dir"], self.config["dataset_path"])
        csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
        csv_files_to_keep = [filename for filename in csv_files if any(artist in filename for artist in files_to_multiartist)]
        print("csv_files: ", csv_files)
        print("csv_files_to_keep: ", csv_files_to_keep)
        datasets = []
        for file in csv_files_to_keep:
            print("FILE: ", file)
            csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], file)
            csvFile = load_dataset("csv", data_files=csv_path, split="train")
            csvFile = csvFile.filter(lambda row: row['lyrics'] is not None and row['artist'] is not None)
            csvFile = csvFile.map(preprocess_lyrics_multiple_artists)
            csvFile = csvFile.select_columns("lyrics")
            datasets.append(csvFile)
            #print("csvFile: ", csvFile[0])
        combined_dataset = concatenate_datasets(datasets)
        print("combined_dataset: ", combined_dataset)
        combined_dataset = combined_dataset.train_test_split(test_size=0.1)
        print("combined_dataset: ", combined_dataset)
        return combined_dataset
