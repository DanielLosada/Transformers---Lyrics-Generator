import os
import re
import zipfile
import pandas as pd

class LyricsDataset():
    def __init__(self, config, artist="Eminem", database="79-musical-genres"):
        self.config = config
        self.artist = artist
        self.database = database
        self.dataset_zip = self.config["dataset_zip"][self.database]

        # Obtain compressed archive 
        archive_name=re.sub(r'^.*?/', '', self.dataset_zip)
        archive_name=re.sub(r'.zip','',archive_name)
        print(archive_name)

        # Check wether selected database matches downloaded database
        if not os.path.exists(os.path.join(config["dataset_path"],archive_name)) or not os.listdir(os.path.join(config["dataset_path"],archive_name)):
            print(os.path.join(self.config["base_dir"], self.dataset_zip))
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.dataset_zip), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.config["base_dir"], self.config["dataset_path"], archive_name))
                print("Successfully extracted the contents of the zip file.")
        else:
            print("The", os.path.join(self.config["base_dir"], self.config["dataset_path"], archive_name), "folder is not empty. Skipping extraction.")

    def load_dataset_single_artist(self):
        csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.artist + ".csv")
        csvFile = self.__load_dataset("csv", data_files=csv_path, split="train")
        # csvFile = csvFile.map(self.__preprocess_lyrics)
        # dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)
        # return dataset

    def load_dataset_multiple_artists(self):
        folder_path = os.path.join(self.config["base_dir"], self.config["dataset_path"])
        csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
        print("csv_files: ", csv_files)
        datasets = []

    # Preprocessing functions 
    def __preprocess_lyrics(self, data):
        print("DATA: ", data)
        # Remove the first line
        data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
        
        # Remove text between square brackets
        data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
        data['lyrics'] = data['lyrics'].strip()
        return data

    # def __preprocess_lyrics_multiple_artists(self, data):
    #     data = __preprocess_lyrics(data)
    #     data['lyrics'] =  data['artist'] + ": " + data['lyrics']
    #     return data
         

    def __load_dataset(self, dataset_extension="csv", dataset_path="", split="train"):
        lyrics = pd.read_csv(dataset_path)
        print(lyrics)
