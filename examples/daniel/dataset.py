import re
import os
import zipfile
import pandas as pd

from datasets import load_dataset, concatenate_datasets, Dataset

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

        if not os.path.exists(self.config["artists_dataset_path"]) or not os.listdir(self.config["artists_dataset_path"]):
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.config["dataset_zip"]), 'r') as zip_ref:
                zip_ref.extractall(self.config["artists_dataset_path"])
                print("Successfully extracted the contents of the zip file.")
            
        for file_name in files_to_delete:
            file_path = os.path.join(self.config["artists_dataset_path"], f"{file_name}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        else:
            print("The " + self.config["artists_dataset_path"] + " folder is not empty. Skipping extraction.")
    
    def load_dataset_single_artist(self):
        csv_path = os.path.join(self.config["base_dir"], self.config["artists_dataset_path"], self.artist + ".csv")
        csvFile = load_dataset("csv", data_files=csv_path, split="train")
        csvFile = csvFile.map(preprocess_lyrics)
        dataset = csvFile.select_columns("lyrics").train_test_split(test_size=0.1)
        return dataset
    
    def load_dataset_multiple_artists(self):
        folder_path = os.path.join(self.config["base_dir"], self.config["artists_dataset_path"])
        csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
        csv_files_to_keep = [filename for filename in csv_files if any(artist in filename for artist in files_to_multiartist)]
        print("csv_files: ", csv_files)
        print("csv_files_to_keep: ", csv_files_to_keep)
        datasets = []
        for file in csv_files_to_keep:
            print("FILE: ", file)
            csv_path = os.path.join(self.config["base_dir"], self.config["artists_dataset_path"], file)
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


class LyricsGenresDataset():
    
    def __init__(self, config, genre = ""):
        self.config = config
        self.genre = genre

        if not os.path.exists(self.config["genres_dataset_path"]) or not os.listdir(self.config["genres_dataset_path"]):
            with zipfile.ZipFile(os.path.join(self.config["base_dir"], self.config["dataset_zip"]), 'r') as zip_ref:
                zip_ref.extractall(self.config["genres_dataset_path"])
                print("Successfully extracted the contents of the zip file.")
        else:
            print("The " + self.config["genres_dataset_path"] + " folder is not empty. Skipping extraction.")

    def load_dataset_genre(self):
        ### Prepare data
        lyrics = pd.read_csv(os.path.join(self.config["genres_dataset_path"],'lyrics-data.csv'))
        lyrics = lyrics[lyrics['language']=='en']
        print("len(lyrics): ", len(lyrics))

        #Only keep popular artists, with genre Rock/Pop and popularity high enough
        artists = pd.read_csv(os.path.join(self.config["genres_dataset_path"],'artists-data.csv'))
        #print("artists: ", artists)
        #artists = artists[(artists['Genres'].isin([self.genre])) & (artists['Popularity']>2)]
        artists = artists[(artists['Genres'].str.contains('Rock', case=False, na=False)) & (artists['Popularity'] > 5)]
        print("len(artists): ", len(artists))
        if not artists.empty:
            df = lyrics.merge(artists[['Artist', 'Genres', 'Link']], left_on='ALink', right_on='Link', how='inner')
            df = df.drop(columns=['ALink','SLink','Link'])
            dataset = Dataset.from_pandas(df).select_columns("Lyric").train_test_split(test_size=0.1)
            print("dataset: ", dataset)
            return dataset
        else:
            print("No artists found for genre: ", self.genre)
            return None


