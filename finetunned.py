!pip install transformers datasets sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import torch

# Upload your dataset (if not already uploaded)
from google.colab import files
uploaded = files.upload()

# Load and preprocess your dataset
df = pd.read_csv("constitution_of_india.csv")
df = df.dropna()
df['input_text'] = "summarize: " + df['article'] + " - " + df['title']
df['target_text'] = df['description']

dataset = Dataset.from_pandas(df[['input_text', 'target_text']])

# Load tokenizer and model (t5-large)
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Tokenize function
def tokenize(batch):
    inputs = tokenizer(batch['input_text'], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(batch['target_text'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['input_text', 'target_text'])

# Set format for PyTorch
tokenized_dataset.set_format(type='torch')

# Training arguments
training_args = TrainingArguments(
    output_dir="./t5-large-legal",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",  # disable W&B
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save model
trainer.save_model("./t5-large-legal")
tokenizer.save_pretrained("./t5-large-legal")

