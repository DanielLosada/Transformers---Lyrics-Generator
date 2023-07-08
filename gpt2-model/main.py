import argparse
import torch
import json
import os
from pynvml import *
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import LyricsDataset
from generator import *

# Load device to use eith GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")
    training_args = TrainingArguments("trainer", per_device_train_batch_size=4, evaluation_strategy="epoch", num_train_epochs=10, save_strategy="epoch", load_best_model_at_end=True)
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
    if(save_name):
        trainer.save_model("./models/" + save_name.replace(" ", "_"))

if __name__ == "__main__":
    os.chdir('/home/paurosci/gits/Transformers---Lyrics-Generator/gpt2-model')
    
    # Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parse the command-line options
    parser = argparse.ArgumentParser(prog='Lyrics generator', description="Trains a lyrics generation model based on GPT-2 architecture.")
    parser.add_argument("-ts", "--trainSingleArtist", dest='train_single_artist', type=str, help="Prepare the dataset of the choosen artist and train the model")
    parser.add_argument("-tm", "--trainMultipleArtists", dest='train_multiple_artists', action="store_true", help="Prepare the dataset of all the artists and train the model")
    parser.add_argument("-tg", "--trainGenre", dest='train_genre', type=str, help="Pass the genre to train the model with songs of that genre")
    
    parser.add_argument("-gs", "--generateSingleArtist", dest='generate_single_artist', type=str, help="Pass the artist name to generate lyrics. Use the same name you used to train it.")
    parser.add_argument("-gm", "--generateMultipleArtist", dest='generate_multiple_artists',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists. Use the same name you used to train it.")
    parser.add_argument("-gg", "--generateGenre", dest='generate_genre',type=str, help="Pass the artist name to generate lyrics with the model trained with multiple artists genre. Use the same name you used to train it.")
    parser.add_argument("-ds", "--datasetSelection", dest='dataset_selection', choices=["genious-lyrics","79-musical-genres"], help="Offers dataset selection between two choices")
    
    parser.add_argument("-gsp", "--singleArtistPerformance", dest='single_artist_performance', help="Computes specified metric to evaluate the specified model")
    parser.add_argument("-gmp", "--multipleArtistsPerformance", dest='multiple_artists_performance', help="Computes specified metric to evaluate the specified model")
    parser.add_argument("-ggp", "--genrePerformance", dest='genre_performance', help="Computes specified metric to evaluate the specified model")

    args = parser.parse_args()

    # Set default arguments
    if args.dataset_selection == None:
        args.dataset_selection="genious-lyrics"

    # TODO: remove this
    args.multiple_artists_performance=True
    args.dataset_selection='79-musical-genres'

    # Training options
    if(args.train_single_artist):
        print("Selected single-artist training: ", args.train_single_artist)
        lyrics_dataset = LyricsDataset(config, args.train_single_artist, args.dataset_selection, args.performance_evaluation)
        lyrics_dataset.load_dataset_single_artist()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_single_artist)
    elif(args.train_multiple_artists):
        print("Selected multi-artist tranining")
        lyrics_dataset = LyricsDataset(config, "multipleArtists", args.dataset_selection, args.performance_evaluation)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, "multipleArtists")
    elif(args.train_genre):
        print("Selected genre tranining")
        lyrics_dataset = LyricsDataset(config, args.train_genre, "79-musical-genres", args.performance_evaluation)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset, args.train_genre)
   
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
    if(args.genre_performance):
        print("Selected genre performance evaluation")
        lyrics_dataset = LyricsDataset(config, args.genre_performance, "79-musical-genres", True)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        train_model(lyrics_dataset, tokenized_dataset)
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.num_sequences = 1
        lyrics_generator_params.max_length = 30
        lyrics_generator = LyricsGenerator(config, args.genre_performance+'_performance', lyrics_generator_params)
    elif(args.multiple_artists_performance):
        print("Selected multiple artists performance evaluation")
        lyrics_dataset = LyricsDataset(config, args.genre_performance, "79-musical-genres", True)
        lyrics_dataset.load_dataset_multiple_artists()
        tokenized_dataset = lyrics_dataset.dataset.map(
            lyrics_dataset.tokenize, batched=True, remove_columns=lyrics_dataset.dataset["train"].column_names
        )
        print("&"*50)
        print('dataset')
        print(lyrics_dataset.dataset['test']['Lyric'][0])
        print("&"*50)
        print('true end lyrics')    
        print(lyrics_dataset.true_end_lyric[0])
        print("&"*50)

        train_model(lyrics_dataset, tokenized_dataset, "multipleArtists_performance")
        lyrics_generator_params = LyricsGeneratorParams
        lyrics_generator_params.num_sequences = 1
        lyrics_generator_params.max_length = 20

        generated_lyrics = []
        lyrics_generator = LyricsGenerator(config, 'multipleArtists_performance', lyrics_generator_params)
        for i in range (len(lyrics_dataset.dataset['test'])):
            if args.dataset_selection == '79-musical-genres':
                lyrics_generator.params.max_length = 20
                lyrics_generator.generate_lyrics(lyrics_dataset.dataset['test']['Lyric'][i])
                generated_lyrics.append(lyrics_generator.generated)
        print('generated_lyrics')
        print(generated_lyrics)
        print("&"*50)




