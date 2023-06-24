import re
import argparse
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import LyricsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def tokenize(element):
    outputs = tokenizer(
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

if __name__ == "__main__":
    #Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(prog='Lyrics generator')
    parser.add_argument("--trainSingleArtist", type=str, help="Prepare the dataset of the choosen artist. And train the model")
    parser.add_argument("--trainMultipleArtists", action="store_true", help="Prepare the dataset of all the artists. And train the model")

    #Parse the command-line arguments
    args = parser.parse_args()

    if(args.trainSingleArtist):
        print("Artist: ", args.trainSingleArtist)
        ly = LyricsDataset(config, args.trainSingleArtist)
        dataset = ly.load_dataset_single_artist()

        context_length = 128
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        tokenizer = tokenizer.to(device)
        tokenizer.pad_token = tokenizer.eos_token

        
        tokenized_datasets = dataset.map(
            tokenize, batched=True, remove_columns=dataset["train"].column_names
        )

        #Load model
        model = AutoModelForCausalLM.from_pretrained(config["model"])
        model.to(device)

        model_size = sum(t.numel() for t in model.parameters())
        print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")

        training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch", num_train_epochs=10)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, device=device)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            #compute_metrics=compute_metrics,
        )

        trainer.train()

    if(args.trainMultipleArtists):
        ly = LyricsDataset(config, args.trainSingleArtist)
        dataset = ly.load_dataset_multiple_artists()


