import os
import re
import zipfile
import math
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer
pd.options.mode.chained_assignment = None  # Remove chained assignment warning

class LyricsDataset():
    def __init__(self, config, filter_field, dataset_id, performance_evaluation_nremovals=None):
        self.config = config
        self.filter_field = filter_field
        self.dataset_id = dataset_id
        self.performance_evaluation_nremovals = performance_evaluation_nremovals
        self.dataset_zip = self.config["dataset_zip"][self.dataset_id]
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.files_to_delete = ["(Scriptonite)", "BTS", "Damso", "Genius English Translations", "Genius Romanizations", "JuL", "Nekfeu", "Oxxxymiron"]
        self.files_to_multiartist = ["50 Cent", "Imagine Dragons", "Justin Bieber", "Taylor Swift", "Queen", "Lil Peep", "Arctic Monkeys", "The Notorious B.I.G.", "Radiohead", "Mac Miller"]
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
        
        if self.dataset_id == 'genius-lyrics':
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
        if self.dataset_id == 'genius-lyrics':
            csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir, self.filter_field + ".csv")
            
            # Check wether performance evaluation needs to be computed
            if(self.performance_evaluation_nremovals):
                csvFile = pd.read_csv(csv_path)
                csvFile = self.__preprocess_lyrics(csvFile)
                self.dataset = self.__split_train_custom_eval(csvFile, test_size=self.config["val_size"])
            else:
                csvFile = load_dataset("csv", data_files=csv_path, split="train")
                csvFile = csvFile.map(self.__preprocess_lyrics_single_artist)
                self.dataset = csvFile.select_columns("lyrics").train_test_split(test_size=self.config["val_size"])

        elif self.dataset_id == '79-musical-genres':
            artists_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/artists-data.csv")
            lyrics_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/lyrics-data.csv")
            artistsCsvFile = pd.read_csv(artists_csv_path)
            lyricsCsvFile = pd.read_csv(lyrics_csv_path)

            # Merge both databases
            csvFile = lyricsCsvFile.merge(artistsCsvFile[['Artist', 'Genres', 'Popularity', 'Link']], left_on='ALink', right_on='Link', how='inner')
            csvFile = self.__preprocess_lyrics_single_artist(csvFile)
            if self.performance_evaluation_nremovals == None:
                self.dataset = Dataset.from_pandas(csvFile).select_columns("Lyric").train_test_split(test_size=self.config["val_size"])
            else:
                self.dataset = self.__split_train_custom_eval(csvFile, test_size=self.config["val_size"])
            

    def load_dataset_multiple_artists(self):
        """Loads several datasets from different artists and stores its cleaned version"""
        if self.dataset_id == 'genius-lyrics':
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
                # Check wether performance evaluation needs to be computed
                if(self.performance_evaluation_nremovals):
                    csvFile = pd.read_csv(csv_path)
                    # csvFile = csvFile.filter(lambda row: row['lyrics'] is not None and row['artist'] is not None)
                    csvFile = self.__preprocess_lyrics_multiple_artists(csvFile)
                    csvFile = csvFile[csvFile.columns.intersection(['lyrics'])]
                    datasets.append(csvFile)
                else:
                    csvFile = load_dataset("csv", data_files=csv_path, split="train")
                    csvFile = csvFile.filter(lambda row: row['lyrics'] is not None and row['artist'] is not None)
                    csvFile = csvFile.map(self.__preprocess_lyrics_multiple_artists)
                    csvFile = csvFile.select_columns("lyrics")
                    datasets.append(csvFile)

            if(self.performance_evaluation_nremovals):
                # Concatenate datasets and store it internally
                csvFile = pd.concat(datasets)
                # shuffle concatenated datasets
                csvFile = csvFile.sample(frac=1)
                csvFile = csvFile.reset_index()
                self.dataset = self.__split_train_custom_eval(csvFile, test_size=self.config["val_size"])
            else:
                # Concatenate datasets and store it internally
                self.dataset = concatenate_datasets(datasets)
                self.dataset = self.dataset.train_test_split(test_size=self.config["val_size"])
            print("combined_dataset: ", self.dataset)
        elif self.dataset_id == '79-musical-genres':
            artists_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/artists-data.csv")
            lyrics_csv_path = os.path.join(self.config["base_dir"], self.config["dataset_path"], self.dataset_dir + "/lyrics-data.csv")
            artistsCsvFile = pd.read_csv(artists_csv_path)
            lyricsCsvFile = pd.read_csv(lyrics_csv_path)
            
            # Merge both databases
            csvFile = lyricsCsvFile.merge(artistsCsvFile[['Artist', 'Genres', 'Popularity', 'Link']], left_on='ALink', right_on='Link', how='inner')
            csvFile = self.__preprocess_lyrics_multiple_artists(csvFile)
            # Modify test dataset in case we want to evaluate performance
            if self.performance_evaluation_nremovals == None:
                self.dataset = Dataset.from_pandas(csvFile).select_columns("Lyric").train_test_split(test_size=self.config["val_size"])
            else:
                self.dataset = self.__split_train_custom_eval(csvFile, test_size=self.config["val_size"])

    def tokenize(self, element):
        """Tokenizes a loaded dataset containing a lyrics section"""
        context_length = 128
        input_batch = []

        if self.dataset_id == 'genius-lyrics':
            outputs = self.tokenizer(
                element["lyrics"],
                truncation=True,
                max_length=context_length,
                #padding="max_length",
                return_overflowing_tokens=True,
                return_length=True,
            )
        elif self.dataset_id == '79-musical-genres':
            outputs = self.tokenizer(
                element["Lyric"],
                truncation=True,
                max_length=context_length,
                #padding="max_length",
                return_overflowing_tokens=True,
                return_length=True,
            )

        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            input_batch.append(input_ids)
        return {"input_ids": input_batch}
    
    def __preprocess_lyrics_single_artist(self, data):
        """Preprocesses lyrics by removing first line and text between square brakets"""
        if self.dataset_id == 'genius-lyrics':
            # Remove the first line
            if data['lyrics'] is not None:
                data['lyrics'] = data['lyrics'].split('\n', 1)[-1]
                
                # Remove text between square brackets
                data['lyrics'] = re.sub(r'\[.*?\]', '', data['lyrics'])
                data['lyrics'] = data['lyrics'].strip()
                data['lyrics'] = re.sub(r'[-+]?(\d+).(\d+)KEmbed', '', data['lyrics'])
                data['lyrics'] = re.sub(r'[-+]?(\d+)KEmbed', '', data['lyrics'])
                data['lyrics'] = re.sub(r'KEmbed', '', data['lyrics'])
                data['lyrics'] = re.sub(r'[-+]?(\d+).(\d+)Embed', '', data['lyrics'])
                data['lyrics'] = re.sub(r'[-+]?(\d+)Embed', '', data['lyrics'])
                data['lyrics'] = re.sub(r'Embed', '', data['lyrics'])
            else:
                data['lyrics'] = ""

        elif self.dataset_id == '79-musical-genres':
            # Select only english songs
            data = data[data['language']=='en']

            # Apply genre filter
            if(self.filter_field):
                data = data[data['Artist'].str.contains(self.filter_field, case=False, na=False)]
            data = data.reset_index()

            
            '''split_data = [j.split() for j in data['Lyric'].split('\n')]
            split_data =  list(filter(None, split_data))
            data['Lyric'] = '\n'.join(' '.join(v) for v in split_data)'''
            data['Lyric'] = data['Lyric'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
            data['Lyric'] = data['Lyric'].apply(lambda x: re.sub(r'\(.*?\)', '', x))
            
            data = data.drop(columns=['ALink','SLink','Link','Popularity'])

        return data
    
    def __preprocess_lyrics(self, data):
        """Preprocesses lyrics by removing first line and text between square brakets"""
        if self.dataset_id == 'genius-lyrics':
            for i in range(len(data['lyrics'])):
                if isinstance(data['lyrics'][i], str):
                    # Remove the first line
                    data['lyrics'][i] = data['lyrics'][i].split('\n', 1)[-1]
                    
                    # Remove text between square brackets
                    data['lyrics'][i] = re.sub(r'\[.*?\]', '', data['lyrics'][i])
                    data['lyrics'][i] = data['lyrics'][i].strip()

                    # Remove double break lines
                    split_data = [j.split() for j in data['lyrics'][i].split('\n')]
                    split_data =  list(filter(None, split_data))
                    data['lyrics'][i] = '\n'.join(' '.join(v) for v in split_data)

                    # Remove last word from lyrics i.e. 1.6KEmbed?
                    data['lyrics'][i] = re.sub(r'[-+]?(\d+).(\d+)KEmbed', '', data['lyrics'][i])
                    data['lyrics'][i] = re.sub(r'[-+]?(\d+)KEmbed', '', data['lyrics'][i])
                    data['lyrics'][i] = re.sub(r'KEmbed', '', data['lyrics'][i])
                    data['lyrics'][i] = re.sub(r'[-+]?(\d+).(\d+)Embed', '', data['lyrics'][i])
                    data['lyrics'][i] = re.sub(r'[-+]?(\d+)Embed', '', data['lyrics'][i])
                    data['lyrics'][i] = re.sub(r'Embed', '', data['lyrics'][i])
                else:
                    data['lyrics'][i] = ""
        elif self.dataset_id == '79-musical-genres':
            pass

        return data

    def __preprocess_lyrics_multiple_artists(self, data):
        """Preprocesses multiple artists lyrics"""
        if self.dataset_id == 'genius-lyrics':
            if self.performance_evaluation_nremovals:
                data = self.__preprocess_lyrics(data)
            else:
                data = self.__preprocess_lyrics_single_artist(data)
            data['lyrics'] =  data['artist'] + ": " + data['lyrics']
        elif self.dataset_id == '79-musical-genres':
            # Select only english songs
            data = data[data['language']=='en']

            # Apply genre filter
            if(self.filter_field == 'multipleArtists'):
                data = data[data['Artist'].str.contains(self.filter_field, case=False, na=False)]
            elif(self.filter_field):
                data = data[(data['Genres'].isin([self.filter_field])) & (data['Popularity']>5)]
            data = data.reset_index()

            # Remove double break lines
            for i in range(len(data['Lyric'])):
                split_data = [j.split() for j in data['Lyric'][i].split('\n')]
                split_data =  list(filter(None, split_data))
                data['Lyric'][i] = '\n'.join(' '.join(v) for v in split_data)
            
            data = data.drop(columns=['ALink','SLink','Link','Popularity'])

        return data

    def __split_train_custom_eval(self, csvFile, test_size):
            """Custom train - eval set split"""
            n_train = math.ceil((1.0 - test_size) * len(csvFile))
            n_test = math.ceil(test_size * len(csvFile))
            n_train, n_test = int(n_train), int(n_test)
            print("n_train: ", n_train, " n_test: ", n_test)
            test_set = csvFile.sample(n = n_test)
            train_set = csvFile.loc[~csvFile.index.isin(test_set.index)]
            train_set = train_set.reset_index()

            # remove last n verses from test set
            test_set = test_set.reset_index()
            if(self.dataset_id=='genius-lyrics'):
                #test_set['lyrics'] = self.__remove_last_verses_from_dataset(test_set['lyrics'], test_set['lyrics'], n=self.performance_evaluation_nremovals)
                test_set['lyrics'] = self.__remove_last_words_from_dataset(test_set['lyrics'], test_set['lyrics'], n=self.performance_evaluation_nremovals)
                train_set = csvFile.loc[~csvFile.index.isin(test_set.index)]
                # Create datasets
                train_dataset = Dataset.from_pandas(train_set).select_columns("lyrics")
                test_dataset = Dataset.from_pandas(test_set).select_columns("lyrics")
            elif(self.dataset_id=='79-musical-genres'):
                #test_set['Lyric'] = self.__remove_last_verses_from_dataset(test_set['Lyric'], test_set['Lyric'], n=self.performance_evaluation_nremovals)
                test_set['Lyric'] = self.__remove_last_words_from_dataset(test_set['Lyric'], test_set['Lyric'], n=self.performance_evaluation_nremovals)
                train_set = csvFile.loc[~csvFile.index.isin(test_set.index)]
                train_dataset = Dataset.from_pandas(train_set).select_columns("Lyric")
                test_dataset = Dataset.from_pandas(test_set).select_columns("Lyric")
                
            # Concatenating train_dataset and test_dataset
            dataset=DatasetDict({'train': train_dataset, 'test': test_dataset})

            return dataset
    
    def __remove_last_verses_from_dataset(self, dataset_candidate, dataset_reference, n):
        """Deletes last n verses from specified dataset"""
        split_true_lyrics_dataset = []
        for i in range(len(dataset_candidate)):
            # Remove last n sentences
            split_dataset = dataset_candidate[i].split('\n')
            split_true_lyrics_dataset = dataset_reference[i].split('\n')
            for j in range(0, n):
                split_dataset.pop()
            for j in range(0, len(split_true_lyrics_dataset)-n):
                split_true_lyrics_dataset.pop(0)
            # Join datasets
            self.true_lyrics_dataset.append('\n'.join(split_true_lyrics_dataset))
            dataset_candidate[i] = '\n'.join(split_dataset)
        return dataset_candidate

    def __remove_last_words_from_dataset(self, dataset_candidate, dataset_reference, n):
        """Deletes last n words from specified dataset"""
        split_true_lyrics_dataset = []
        for i in range(len(dataset_candidate)):
            # Remove last n words
            split_dataset = [j.split() for j in dataset_candidate[i].split('\n')]
            split_true_lyrics_dataset = [j.split() for j in dataset_reference[i].split('\n')]
            word_count=0
            for j in reversed(range(0,len(split_dataset))):
                if split_dataset[j] != '[]':
                    for k in reversed(range(len(split_dataset[j]))):
                        if word_count >= n:
                            split_true_lyrics_dataset[j].pop(k)
                        else:
                            #print("removed word count: ", word_count, " removed word: ", split_dataset[j][k])
                            split_dataset[j].pop(k)
                            word_count+=1
                    else:
                        continue
            if(word_count != n):
                print('Only removed '  + str(word_count) + ' words from data')
                print('test set reference ' + i + ' length was: '  + str(dataset_reference[i]) + '...')
            # Join datasets
            split_dataset =  list(filter(None, split_dataset))
            split_true_lyrics_dataset =  list(filter(None, split_true_lyrics_dataset))
            self.true_lyrics_dataset.append('\n'.join(' '.join(v) for v in split_true_lyrics_dataset))
            dataset_candidate[i] = '\n'.join(' '.join(v) for v in split_dataset)
        return dataset_candidate