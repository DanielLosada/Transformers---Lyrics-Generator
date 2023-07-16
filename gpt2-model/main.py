import argparse
import torch
import json
import os
import wandb

from datetime import datetime
from pynvml import *
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import LyricsDataset
from generator import LyricsGeneratorParams, LyricsGenerator
from performance import *

# Load device to use eith GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#### MODEL TRAINING FUNCTIONS #### 
def initialise_wandb_project(config, model_name):
    os.environ["WANDB_API_KEY"] = config["wandb"]["wandb_api_key"]
    wandb.init(project="Lyrics Generator")
    wandb.run.name = f'{model_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

def print_gpu_utilization():
    """Prints GPU usage while training"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    """Prints training summary results"""
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # print_gpu_utilization()

def train_model(dataset, tokenized_dataset, save_name=''):
    initialise_wandb_project(config, save_name.replace(" ", "_"))

    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")
    training_args = TrainingArguments("trainer", report_to="wandb", learning_rate=5e-6, run_name=f'{save_name.replace(" ", "_")}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',per_device_train_batch_size=4, evaluation_strategy="epoch", num_train_epochs=config["epochs"], save_strategy="epoch", load_best_model_at_end=True)
    data_collator = DataCollatorForLanguageModeling(dataset.tokenizer, mlm=False)
    trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=dataset.tokenizer,
            #compute_metrics=compute_metrics,
        )
    # print_gpu_utilization()
    trainer.train()
    #print_summary(result)
    wandb.finish()
    if(save_name):
        trainer.save_model("./models/" + save_name.replace(" ", "_"))

if __name__ == "__main__":
    # TODO: remove this
    # os.chdir('/Users/prosci/gits/Transformers---Lyrics-Generator/gpt2-model')

    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parse the command-line options
    parser = argparse.ArgumentParser(prog='Lyrics generator', description="Trains a lyrics generation model based on GPT-2 architecture.")
    parser.add_argument("-ts", "--trainSingleArtist", dest='train_single_artist', type=str, help="Prepare the dataset of the choosen artist and train the model")
    parser.add_argument("-tm", "--trainMultipleArtists", dest='train_multiple_artists', action="store_true", help="Prepare the dataset of all the artists and train the model")
    parser.add_argument("-tg", "--trainGenre", dest='train_genre', type=str, help="Pass the genre to train the model with songs of that genre")
    
    parser.add_argument("-gs", "--generateSingleArtist", dest='generate_single_artist', type=str, help="Pass the artist name to generate lyrics. Use the same name you used to train it.")
    parser.add_argument("-gm", "--generateMultipleArtists", dest='generate_multiple_artists',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists. Use the same name you used to train it.")
    parser.add_argument("-gg", "--generateGenre", dest='generate_genre',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists genre. Use the same name you used to train it.")
    parser.add_argument("-ds", "--datasetSelection", dest='dataset_selection', choices=["genious-lyrics","79-musical-genres"], default = "genious-lyrics", help="Offers dataset selection between two choices")
    
    parser.add_argument("-sp", "--singleArtistPerformance", nargs=3, dest='single_artist_performance', help="Computes the metric to evaluate the single-artist model")
    parser.add_argument("-mp", "--multipleArtistsPerformance", nargs=3, dest='multiple_artists_performance', help="Computes the metric to evaluate the multiple-artist model")
    parser.add_argument("-gp", "--genrePerformance", nargs=3, dest='genre_performance', help="Computes the metric to evaluate the genre model")

    args = parser.parse_args()

    # TODO: remove this
    # args.single_artist_performance=['Bob Dylan', 'True', '20']
    # args.dataset_selection = '79-musical-genres'
    # args.multiple_artists_performance=['Eminem10', 'True', '20']
    # args.train_multiple_artists = True
    # args.genre_performance = ['Rock', 'True', '31']
    
    # Training options
    if(args.train_single_artist):
        print("Selected single-artist training: ", args.train_single_artist)
        lyrics_dataset = LyricsDataset(config, args.train_single_artist, args.dataset_selection)
        lyrics_dataset.load_dataset_single_artist()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_single_artist + '_' + args.dataset_selection)
    elif(args.train_multiple_artists):
        print("Selected multi-artist tranining")
        lyrics_dataset = LyricsDataset(config, "multipleArtists", args.dataset_selection)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, "multipleArtists_" + args.dataset_selection)
    elif(args.train_genre):
        print("Selected genre tranining")
        lyrics_dataset = LyricsDataset(config, args.train_genre, "79-musical-genres")
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_genre + "_79-musical-genres")
   
    # Generation options
    if(args.generate_single_artist):
        print("Selected single-artist generation: ", args.generate_single_artist)
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        lyrics_generator = LyricsGenerator(config, args.generate_single_artist, lyrics_generator_params)
        lyrics_generator.generate_lyrics(initial_prompt="My name is")
    elif(args.generate_multiple_artists):
        print("Selected multiple-artist generation")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 10
        initial_prompt="You are"
        lyrics_generator = LyricsGenerator(config, "multipleArtists", lyrics_generator_params)
        lyrics_generator.generate_lyrics(args.generate_multiple_artists + ': ' + initial_prompt)
    elif(args.generate_genre):
        print("Selected multiple-artist genre generation")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.max_length = 30
        initial_prompt="You are"
        lyrics_generator = LyricsGenerator(config, "multipleArtistsGenre", lyrics_generator_params)
        lyrics_generator.generate_lyrics(args.generate_genre + ': ' + initial_prompt)

    # Performance evaluation options
    if (args.single_artist_performance):
        print("Selected single artist performance evaluation")
        artist = args.single_artist_performance[0]
        train = args.single_artist_performance[1]
        n_words = int(args.single_artist_performance[2])

        # Train the model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, artist, args.dataset_selection, performance_evaluation_nremovals=n_words)
            lyrics_dataset.load_dataset_single_artist()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, artist.replace(" ", "_") + "_performance")

            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            if args.dataset_selection == 'genious-lyrics':
                dataset = lyrics_dataset.dataset['test']['lyrics']
            elif args.dataset_selection == '79-musical-genres':
                dataset = lyrics_dataset.dataset['test']['Lyric']

            for i in range(len(dataset)):
                test_trimmed_lyrics.extend([str(dataset[i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset

            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("./models/" + artist.replace(" ", "_") + "_performance"):
                os.makedirs("./models/" + artist.replace(" ", "_") + "_performance")
            with open("./models/" + artist.replace(" ", "_") + "_performance/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)

        # Initialise wandb logging
        initialise_wandb_project(config, artist.replace(" ", "_")+'_performance')

        # Evaluate the model
        performance_data = compute_bleu_metric(config, n_words, artist)

    elif(args.multiple_artists_performance):
        print("Selected multiple artists performance evaluation")
        artist_generation = args.multiple_artists_performance[0]
        train = args.multiple_artists_performance[1]
        n_words = int(args.multiple_artists_performance[2])
        artist = 'multipleArtists'

        # Train the model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, 'multipleArtists', args.dataset_selection, performance_evaluation_nremovals=n_words)
            lyrics_dataset.load_dataset_multiple_artists()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, artist + "_performance")
            
            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            if args.dataset_selection == 'genious-lyrics':
                dataset = lyrics_dataset.dataset['test']['lyrics']
            elif args.dataset_selection == '79-musical-genres':
                dataset = lyrics_dataset.dataset['test']['Lyric']

            for i in range(len(dataset)):
                test_trimmed_lyrics.extend([str(dataset[i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset

            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("./models/" + artist + "_performance"):
                print("We dont have it")
                os.makedirs("./models/" + artist + "_performance")
            with open("./models/" + artist + "_performance/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)
       
        # Initialise wandb logging
        initialise_wandb_project(config, artist.replace(" ", "_")+'_performance')

        # Evaluate the model
        performance_data = compute_bleu_metric(config, n_words, artist, artist_generation)


    elif(args.genre_performance):
        print("Selected genre performance evaluation")
        genre = args.genre_performance[0]
        train = args.genre_performance[1]
        n_words = int(args.genre_performance[2])

        # Train the model
        if(train == 'True'):
            lyrics_dataset = LyricsDataset(config, genre, "79-musical-genres", performance_evaluation_nremovals=n_words)
            lyrics_dataset.load_dataset_multiple_artists()
            tokenized_dataset = lyrics_dataset.dataset.map(
                lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
            )
            train_model(lyrics_dataset, tokenized_dataset, genre + "_performance")
            
            # Store test lyrics in json file in order to prevent training again
            test_lyrics = {}
            test_trimmed_lyrics = []
            for i in range(len(lyrics_dataset.dataset['test']['Lyric'])):
                test_trimmed_lyrics.extend([str(lyrics_dataset.dataset['test']['Lyric'][i])])
            test_lyrics['test_trimmed_lyrics'] = test_trimmed_lyrics
            test_lyrics['test_true_lyrics'] = lyrics_dataset.true_lyrics_dataset
            with open("./models/" + genre + "_performance/lyrics_test.json","w") as f:
                json.dump(test_lyrics, f)
        
        # Initialise wandb logging
        initialise_wandb_project(config, genre.replace(" ", "_")+'_performance')

        # Evaluate the model
        performance_data = compute_bleu_metric(config, n_words, genre)


