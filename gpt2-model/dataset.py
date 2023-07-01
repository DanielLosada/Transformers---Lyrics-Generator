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
        self.dataset=""

        # Obtain selected dataset specific folder
        archive_name=re.sub(r'^.*?/', '',self.dataset_zip)
        archive_name=re.sub(r'.zip','',archive_name)
        self.dataset_dir = archive_name

        # Check wether selected database matches downloaded database
        if not os.path.exists(os.path.join(config["dataset_path"],self.dataset_dir)) or not os.listdir(os.path.join(config["dataset_path"],self.dataset_dir)):
            print(os.path.join(self.config["base_dir"], self.dataset_zip))
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.dataset_zip), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir))
                print("Successfully extracted the contents of the zip file.")
        else:
            print("The", os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir), "folder is not empty. Skipping extraction.")

    def load_dataset_single_artist(self):
        """Loads a dataset from a specific artist and stores its cleaned version"""
        csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir, self.artist + ".csv")
        csvFile = load_dataset("csv", data_files=csv_path, split="train")
        csvFile = csvFile.map(self.__preprocess_lyrics)
        self.dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)

    def load_dataset_multiple_artists(self):
        """Loads several datasets from different artists and stores its cleaned version"""
        folder_path = os.path.join(self.config["base_dir"], self.config["dataset_path"])
        csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
        print("csv_files: ", csv_files)
        datasets = []

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

        # Remove the first line
        data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
        
        # Remove text between square brackets
        data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
        data['lyrics'] = data['lyrics'].strip()
        
        # TODO: Remove last word from lyrics i.e. 1.6KEmbed?
        # res = re.search(r'[-+]?(\d+)KEmbed', data['lyrics'])

        return data
