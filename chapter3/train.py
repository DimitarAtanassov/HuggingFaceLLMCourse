# Load the data set
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare for training

# Remove the columns corresponding to values the model does not expect
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])

# Rename the column 'label' to 'labels'
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

# Set the format of the datasets so they return PyTorch tensors instead of lists
tokenized_datasets.set_format('torch')

# Updated Columns
tokenized_datasets['train'].column_names

# Define the dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# Sanity check for no mistakes in data processing
for batch in train_dataloader:
    break

{k: v.shape for k, v in batch.items()}

# Init model to finetune 
from transformers import AutoModelForSequenceClassification

# num_labels - Number of labels to use in the last layer added to the model, typically for a classification task.
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Sanity check before training on a single batch
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# Init optimizer
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Init learning rate scheduler
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# The Training loop w/ accelerator
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.to(accelerator.device)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluation Loop
import evaluate
import torch

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute()) 