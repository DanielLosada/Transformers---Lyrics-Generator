import re
import argparse
import json
import torch

from pynvml import *
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import LyricsDataset
from transformers import GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def remove_consecutive_duplicates(arr, max_repeat):
    results = []
    if len(arr) >= max_repeat:
        current_word = arr[0]
        results.append(arr[0])
        count = 1
        for x in range(1, len(arr)):
            if arr[x] != current_word:
                current_word = arr[x]
                count = 0
            if current_word == arr[x]:
                count += 1
            if max_repeat >= count:
                results.append(arr[x])
        return results
    else:
        return arr
            
def post_process(output_sequences):
    predictions = []
    generated_sequences = []

    max_repeat = 2

    # decode prediction
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        generated_sequences.append(text.strip())
    for i, g in enumerate(generated_sequences):
        res = str(g).replace('\n\n\n', '\n').replace('\n\n', '\n')
        lines = res.split('\n')
        print(lines)
        lines = remove_consecutive_duplicates(lines, max_repeat)        
        predictions.append('\n'.join(lines))

    return predictions


def tokenize(element):
    context_length = 128
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

def train_model(tokenized_datasets):
        
        #Load model
        print("Loading model")
        model = AutoModelForCausalLM.from_pretrained(config["model"])
        model.to(device)
        print_gpu_utilization()
        model_size = sum(t.numel() for t in model.parameters())
        print(f"{config['model']} size: {model_size/1000**2:.1f}M parameters")

        training_args = TrainingArguments("trainer", per_device_train_batch_size=4, evaluation_strategy="epoch", num_train_epochs=10, save_strategy="epoch", load_best_model_at_end=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            #compute_metrics=compute_metrics,
        )
        print_gpu_utilization()
        trainer.train()
        #print_summary(result)
        trainer.save_model("./models/" + args.trainSingleArtist.replace(" ", "_"))
        
def generate():
    
    start = "Nigga,"
    num_sequences =  3
    min_length =  1
    max_length =   5
    temperature = 1.51 
    top_p = 0.9 
    top_k = 30 
    repetition_penalty =  1.0

    encoded_prompt = tokenizer(start, add_special_tokens=False, return_tensors="pt").input_ids
    encoded_prompt.to(device)
    print("encoded_prompt: ", encoded_prompt)

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained("./models/" + args.generateSingleArtist.replace(" ", "_") + "/")
    output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=max_length,
                    min_length=min_length,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    do_sample=True,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_sequences)
    print("output_sequences: ", output_sequences)
    generated = post_process(output_sequences)
    for text in generated:
        print(text)



if __name__ == "__main__":
    print_gpu_utilization()
    #Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(prog='Lyrics generator')
    parser.add_argument("--trainSingleArtist", type=str, help="Prepare the dataset of the choosen artist. And train the model")
    parser.add_argument("--trainMultipleArtists", action="store_true", help="Prepare the dataset of all the artists. And train the model")
    parser.add_argument("--generateSingleArtist", type=str, help="Pass the artist name to generate lyrics. Use the same name you used to train it.")

    #Parse the command-line arguments
    args = parser.parse_args()

    if(args.trainSingleArtist):
        print("Artist: ", args.trainSingleArtist)
        ly = LyricsDataset(config, args.trainSingleArtist)
        dataset = ly.load_dataset_single_artist()

        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        tokenizer.pad_token = tokenizer.eos_token

        tokenized_datasets = dataset.map(
            tokenize, batched=True, remove_columns=dataset["train"].column_names
        )

        train_model(tokenized_datasets)
        
    if(args.trainMultipleArtists):
        ly = LyricsDataset(config, args.trainSingleArtist)
        dataset = ly.load_dataset_multiple_artists()
    
    if(args.generateSingleArtist):
        #generation_config = GenerationConfig.from_pretrained("./models/" + args.generateSingleArtist.replace(" ", "_") + "/")
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        tokenizer.pad_token = tokenizer.eos_token
        generate()
        pass



