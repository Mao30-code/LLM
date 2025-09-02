#Import information, tokens Hugging Face
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the dataset from CSV, for this now that Is an small exercise I used a dataset of 20, and 10 for validation
dataset = load_dataset('csv', data_files={
    'train': 'fac_train.csv',
    'validation': 'fac_validation.csv'
})

# Here I load then tokenizer and model (BART base) for model clasification on 3
#chuncking not needed for small dataset, 
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples["Description"], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Renombrar columna Status a label para Hugging Face Trainer
tokenized_datasets = tokenized_datasets.rename_column("Status", "label")


#Here I add model's specifications
training_args = TrainingArguments(
    num_train_epochs=3,                           
    save_strategy="epoch",                          
    logging_dir='./logs', 
    output_dir="./invoice_model",                                      
    per_device_train_batch_size=16,                
    per_device_eval_batch_size=10                                       
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

#here tran and save the model.
trainer.train()

model.save_pretrained("./invoice_model")
tokenizer.save_pretrained("./invoice_model")