import os
import re
import zipfile
import csv
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

class LyricsDataset():
    def __init__(self, config, artist, dataset_id):
        self.config = config
        self.artist = artist
        self.dataset_id = dataset_id
        self.dataset_zip = self.config["dataset_zip"][self.dataset_id]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.files_to_delete = ["(Scriptonite)", "BTS", "Damso", "Genius English Translations", "Genius Romanizations", "JuL", "Nekfeu", "Oxxxymiron"]
        self.files_to_multiartist = ["Eminem10", "Justin Bieber10"]
        #self.files_to_multiartist = ["50 Cent", "Imagine Dragons", "Justin Bieber", "Taylor Swift", "Queen", "Lil Peep", "Arctic Monkeys", "The Notorious B.I.G.", "Radiohead", "Mac Miller"]
        self.dataset=""

        # Obtain selected dataset specific folder
        archive_name=re.sub(r'^.*?/', '',self.dataset_zip)
        archive_name=re.sub(r'.zip','',archive_name)
        self.dataset_dir = archive_name

        # Check wether selected database matches downloaded database
        if not os.path.exists(os.path.join(config["dataset_path"],self.dataset_dir)) or not os.listdir(os.path.join(config["dataset_path"],self.dataset_dir)):
            print(os.path.join(self.config["base_dir"], self.dataset_zip))
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.dataset_zip), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.config["base_dir"], self.config["dataset_path"]))
                print("Successfully extracted the contents of the zip file.")
        else:
            print("The", os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir), "folder is not empty. Skipping extraction.")
        
        # Remove non english authors from dataset 
        for file_name in self.files_to_delete:
            file_path = os.path.join(self.config["dataset_path"], self.dataset_dir,f"{file_name}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")

    def load_dataset_single_artist(self):
        """Loads a dataset from a specific artist and stores its cleaned version"""
        if self.dataset_id == 'genious-lyrics':
            csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir, self.artist + ".csv")
            csvFile = load_dataset("csv", data_files=csv_path, split="train")
            csvFile = csvFile.map(self.__preprocess_lyrics)
            self.dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)
        
        elif self.dataset_id == '79-musical-genres':
            pass
            # artists_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/artists-data.csv")
            # lyrics_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/lyrics-data.csv")
   
            # artistsCsvFile = load_dataset("csv", data_files=artists_csv_path, split="train")
            # lyricsCsvFile = load_dataset("csv", data_files=lyrics_csv_path, split="train")
            # csvFile = lyricsCsvFile.merge(artistsCsvFile[['Artist', 'Genre', 'Link']], left_on='ALink', right_on='Link', how='inner')
            # csvFile = csvFile.drop(columns=['ALink','SLink','Idiom','Link'])
            # csvFile = csvFile.map(self.__preprocess_lyrics)

    def load_dataset_multiple_artists(self):
        """Loads several datasets from different artists and stores its cleaned version"""
        folder_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir)
        csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
        csv_files_to_keep = [filename for filename in csv_files if any(artist in filename for artist in self.files_to_multiartist)]
        print("csv_files: ", csv_files)
        print("csv_files_to_keep: ", csv_files_to_keep)
        datasets = []

        # Append preprocess selected multiple artists datasets
        for file in csv_files_to_keep:
            print("FILE: ", file)
            csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir, file)
            csvFile = load_dataset("csv", data_files=csv_path, split="train")
            csvFile = csvFile.filter(lambda row: row['lyrics'] is not None and row['artist'] is not None)
            csvFile = csvFile.map(self.__preprocess_lyrics_multiple_artists)
            csvFile = csvFile.select_columns("lyrics")
            datasets.append(csvFile)
        
        # Concatenate datasets and store it internally
        self.dataset = concatenate_datasets(datasets)
        print("combined_dataset: ", self.dataset)
        self.dataset = self.dataset.train_test_split(test_size=0.1)
        print("combined_dataset: ", self.dataset)

    def tokenize(self, element):
        """Tokenizes a loaded dataset containing a lyrics section"""
        context_length = 128
        outputs = self.tokenizer(
            element["lyrics"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    
    def __preprocess_lyrics(self, data):
        """Preprocesses lyrics by removing first line and text between square brakets"""
        if self.dataset_id == 'genious-lyrics':
            # Remove the first line
            data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
            
            # Remove text between square brackets
            data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
            data['lyrics'] = data['lyrics'].strip()
            
            # TODO: Remove last word from lyrics i.e. 1.6KEmbed?
            # res = re.search(r'[-+]?(\d+)KEmbed', data['lyrics'])
        elif self.dataset_id == '79-musical-genres':
            pass
            # data['lyrics'] = data[data['Idiom']=='ENGLISH']
            # print(data)
        return data
    
    def __preprocess_lyrics_multiple_artists(self, data):
        """Preprocesses multiple artists lyrics by removing first line and text between square brakets"""
        data = self.__preprocess_lyrics(data)
        data['lyrics'] =  data['artist'] + ": " + data['lyrics']

        return data

