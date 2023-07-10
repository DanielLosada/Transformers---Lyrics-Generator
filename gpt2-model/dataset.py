import os
import re
import zipfile
import math
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer

class LyricsDataset():
    def __init__(self, config, filter_field, dataset_id, performance_evaluation_nverses=None):
        self.config = config
        self.filter_field = filter_field
        self.dataset_id = dataset_id
        self.performance_evaluation_nverses = performance_evaluation_nverses
        self.dataset_zip = self.config["dataset_zip"][self.dataset_id]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.files_to_delete = ["(Scriptonite)", "BTS", "Damso", "Genius English Translations", "Genius Romanizations", "JuL", "Nekfeu", "Oxxxymiron"]
        self.files_to_multiartist = ["Eminem10", "Justin Bieber10"]
        # TODO: Remove this
        #self.files_to_multiartist = ["50 Cent", "Imagine Dragons", "Justin Bieber", "Taylor Swift", "Queen", "Lil Peep", "Arctic Monkeys", "The Notorious B.I.G.", "Radiohead", "Mac Miller"]
        self.dataset=""
        self.true_lyrics_dataset=[]

        # Obtain selected dataset specific folder
        archive_name=re.sub(r'^.*?/', '',self.dataset_zip)
        archive_name=re.sub(r'.zip','',archive_name)
        self.dataset_dir = archive_name
        print(archive_name)

        # Check wether selected database matches downloaded database
        if not os.path.exists(os.path.join(config["dataset_path"],self.dataset_dir)) or not os.listdir(os.path.join(config["dataset_path"],self.dataset_dir)):
            print(os.path.join(self.config["base_dir"], self.dataset_zip))
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.dataset_zip), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_id))
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
            #TODO: preprocess data differently
            #csvFile = load_dataset("csv", data_files=csv_path, split="train")
            #csvFile = csvFile.map(self._preprocess_lyrics)
            csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir, self.filter_field + ".csv")
            csvFile = pd.read_csv(csv_path)
            csvFile = self._preprocess_lyrics(csvFile)

            # Check wether performance evaluation needs to be computed
            if(self.performance_evaluation_nverses):
                self.dataset = self._split_train_custom_eval(csvFile, test_size=0.1)
            else:
                self.dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)
        elif self.dataset_id == '79-musical-genres':
            pass

    def load_dataset_multiple_artists(self):
        """Loads several datasets from different artists and stores its cleaned version"""
        if self.dataset_id == 'genious-lyrics':
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
                csvFile = csvFile.map(self._preprocess_lyrics_multiple_artists)
                csvFile = csvFile.select_columns("lyrics")
                datasets.append(csvFile)
            
            # Concatenate datasets and store it internally
            self.dataset = concatenate_datasets(datasets)
            self.dataset = self.dataset.train_test_split(test_size=0.1)
            print("combined_dataset: ", self.dataset)
        
        elif self.dataset_id == '79-musical-genres':
            artists_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/artists-data.csv")
            lyrics_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/lyrics-data.csv")
            artistsCsvFile = pd.read_csv(artists_csv_path)
            lyricsCsvFile = pd.read_csv(lyrics_csv_path)
            
            # Merge both databases
            csvFile = lyricsCsvFile.merge(artistsCsvFile[['Artist', 'Genres', 'Popularity', 'Link']], left_on='ALink', right_on='Link', how='inner')
            csvFile = self._preprocess_lyrics_multiple_artists(csvFile)

            # Modify test dataset in case we want to evaluate performance
            if self.performance_evaluation_nverses == None:
                self.dataset = Dataset.from_pandas(csvFile).select_columns("Lyric").train_test_split(test_size=0.1)
            else:
                # TODO
                pass

    def tokenize(self, element):
        """Tokenizes a loaded dataset containing a lyrics section"""
        context_length = 128
        input_batch = []

        if self.dataset_id == 'genious-lyrics':
            outputs = self.tokenizer(
                element["lyrics"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
        elif self.dataset_id == '79-musical-genres':
            outputs = self.tokenizer(
                element["Lyric"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )

        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    
    def _preprocess_lyrics(self, data):
        """Preprocesses lyrics by removing first line and text between square brakets"""
        if self.dataset_id == 'genious-lyrics':
            # TODO: preprocess data differently to allow performance evaluation. Remove once confirmed
            # # Remove the first line
            # data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
            # # Remove text between square brackets
            # data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
            # data['lyrics'] = data['lyrics'].strip()
            
            for i in range(len(data['lyrics'])):
                # Remove the first line
                data['lyrics'][i] = data['lyrics'][i].split('\n', 1)[-1]
                # Remove text between square brackets
                data['lyrics'][i] = re.sub(r'\[.*?\]', '', data['lyrics'][i])
                data['lyrics'][i] = data['lyrics'][i].strip()
                # Remove last word from lyrics i.e. 1.6KEmbed?
                data['lyrics'][i] = re.sub(r'[-+]?(\d+).(\d+)KEmbed', '', data['lyrics'][i])
                data['lyrics'][i] = re.sub(r'[-+]?(\d+)KEmbed', '', data['lyrics'][i])
                data['lyrics'][i] = re.sub(r'KEmbed', '', data['lyrics'][i])
                data['lyrics'][i] = re.sub(r'[-+]?(\d+).(\d+)Embed', '', data['lyrics'][i])
                data['lyrics'][i] = re.sub(r'[-+]?(\d+)Embed', '', data['lyrics'][i])
                data['lyrics'][i] = re.sub(r'Embed', '', data['lyrics'][i])

        elif self.dataset_id == '79-musical-genres':
            pass

        return data

    def _preprocess_lyrics_multiple_artists(self, data):
        """Preprocesses multiple artists lyrics by removing first line and text between square brakets"""
        if self.dataset_id == 'genious-lyrics':
            data = self._preprocess_lyrics(data)
            data['lyrics'] =  data['artist'] + ": " + data['lyrics']
        elif self.dataset_id == '79-musical-genres':
            # Select only english songs
            data = data[data['language']=='en']

            # Apply genre filter
            if(self.filter_field):
                data = data[(data['Genres'].str.contains(self.filter_field, case=False, na=False)) & (data['Popularity'] > 5)]
            
            data = data.drop(columns=['ALink','SLink','Link','Popularity'])
            #TODO: Remove this
            data = data.drop(data.index[2:-1])
        return data

    def _split_train_custom_eval(self, csvFile, test_size):
            """Custom train - eval set split"""
            n_train = math.ceil((1.0 - test_size) * len(csvFile))
            n_test = math.ceil(test_size * len(csvFile))
            n_train, n_test = int(n_train), int(n_test)
            print("n_train: ", n_train, " n_test: ", n_test)
            test_set = csvFile.sample(n = n_test)
            test_set = test_set.reset_index()

            # remove last n verses from test set
            test_set = self._remove_last_verses_from_dataset(test_set, test_set, n=self.performance_evaluation_nverses)
            train_set = csvFile.loc[~csvFile.index.isin(test_set.index)]
            train_set = train_set.reset_index()

            # Create datasets
            if(self.dataset_id=='genious-lyrics'):
                train_dataset = Dataset.from_pandas(train_set).select_columns("lyrics")
                test_dataset = Dataset.from_pandas(test_set).select_columns("lyrics")
            elif(self.dataset_id=='79-musical-genres'):
                train_dataset = Dataset.from_pandas(train_set).select_columns("Lyric")
                test_dataset = Dataset.from_pandas(test_set).select_columns("Lyric")
                
            # Concatenating train_dataset and test_dataset
            dataset=DatasetDict({'train': train_dataset, 'test': test_dataset})

            return dataset
    
    def _remove_last_verses_from_dataset(self, dataset, dataset2, n):
        """Deletes last n words from specified lyrics dataset"""
        if self.dataset_id == 'genious-lyrics':
            split_true_lyrics_dataset = []
            for i in range(len(dataset['lyrics'])):
                # Remove last n sentences
                split_dataset = dataset['lyrics'][i].split('\n')
                split_true_lyrics_dataset = dataset2['lyrics'][i].split('\n')
                for j in range(0, n):
                    split_dataset.pop()
                for j in range(0, len(split_true_lyrics_dataset)-n):
                    split_true_lyrics_dataset.pop(0)
                # Join datasets
                self.true_lyrics_dataset.append('\n'.join(split_true_lyrics_dataset))
                dataset['lyrics'][i] = '\n'.join(split_dataset)
            return dataset
        
        elif self.dataset_id == '79-musical-genres':
            # TODO
            pass
        return dataset

