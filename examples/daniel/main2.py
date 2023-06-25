import csv

from sklearn.model_selection import train_test_split
from transformers import TextDataset,DataCollatorForLanguageModeling, AutoTokenizer, Trainer, TrainingArguments,AutoModelWithLMHead


def load_csv_file(file_path):
        lyrics = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the first row as the header
            for row in reader:
                lyrics.append(row[-1])     
        return lyrics

def build_text_files(lyrics, dest_path):
    f = open(dest_path, 'w', encoding='utf-8')
    data = ''
    for texts in lyrics:
        texts = texts.strip()
        data += texts + "\n"
    f.write(data)

data = load_csv_file("./archive/AAP Rocky.csv")

train, test = train_test_split(data,test_size=0.15) 

build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))

tokenizer = AutoTokenizer.from_pretrained("gpt2")

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=32)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=32)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

model = AutoModelWithLMHead.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print("Before train")
trainer.train()