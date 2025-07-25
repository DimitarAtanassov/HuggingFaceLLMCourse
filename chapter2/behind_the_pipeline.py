from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

"""
Preprocessing with a tokenizer
Tokenizer is responsible for:
- Splitting the input into words, subwords, or symbols (like punctuation) that are called tokens
- Mapping each token to an integer
- Adding additional inputs that may be useful to the model
"""

# 1.) Create a tokenizer for preprocessing, this needs to be done in the same way as when the model was pretrained.
print(f'\nCreating a tokenizer for preprocessing:')

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(f'\nTokenizer downloaded: \n{tokenizer}\n')

"""
Once we have our tokenizer, we can directly pass our sentences to it and we'll get back a dictionary that's ready to feed to our model.
The only thing left to do is to convert the list of input IDs to tensors

Transformer models only accept tensors as input.
- If this is your first time hearing about tensors, you can think of them as NumPy arrays instead.
- A NumPy array can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions.
"""

print(f'\nTokenizing\n')

# We cannot feed raw inputs into our model
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]

for idx, input_str in enumerate(raw_inputs):
    print(f'Input String {idx + 1}: {input_str}\n')

# This tokenizes our raw input strings so we can feed them to our model
model_inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # To specify the type of tensors we want to get back (PyTorch or plain NumPy), we use the 'return_tensors' argument

# The output itself is a dictionary containing two keys, input_ids and attention_mask
# input_ids contain two rows of integers (one for each sentence in the input) that are the unique identifiers of the tokens in each sentence.
print(f'Tokenized Input Strings:\n {model_inputs}\n')

for idx, input_str in enumerate(raw_inputs):
    print(f'Input String {idx + 1}: {input_str}')
    print(f'Input String Tokenized: {model_inputs.input_ids[idx]}\n')

# 2.) Going through the model
print("Downloading a Model\n")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
print(f'Model we are downloading: {checkpoint}\n')

"""
The architecture below contains only the base Transformer module.
So, given some inputs, it outputs what we will call hidden states, also known as 'features'
"""
model = AutoModel.from_pretrained(checkpoint) # This downloads the model
print(f'Downloaded Model: {model}\n')

model_outputs = model(**model_inputs)

print(f'Model Outputs: {model_outputs}\n')

""" 
Hidden States are high-dimensional vectors
High-dimensional vectors usually have 3 dimensions
Hidden states can be useful on their own, but they are usually inputs to another part of the model, known as the 'head'
"""
print(f'Model Outputs hidden state: {model_outputs.last_hidden_state.shape}')

# Model Heads: Making sense out of numbers
"""
The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.

The output of the Transformer model is sent directly to the model head to be processed.

The embeddings layer of the model converts each input ID in the tokenized input into a vector that represents the associated token.
The subsequent layers manipulate those vectors using the attention mechansim to produce the final representation of the sentences.
"""

# Example:
# We will need a model with a sequence classification head (inorder to be able to classify the sentences as positive or negative). So, we will not actually use the AutoModel class, but the AutoModelForSequenceClassification:

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs = model(**model_inputs)

# The shape of our outputs, the dimensionality will be much lower:
# The model head takes as input the high-dimensional vectors we saw before, and ouputs vectors containing two values (one per label)
print(f'\nSequence Classification Model Outputs:\n {outputs.logits.shape}') # The result we get from our model is of shape 2x2


# Postprocessing the output:
"""
The values we get as output from our model don't necessarily make sense by themselves.
  This is because the outputs from our model are NOT probabilities but 'logits'

  Logits: The raw unnormalized scores outputted by the last layer of the model.
  - To be converted to probabilities, they need to go through a SoftMax layer
"""

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # these are recognizable probability scores
print('\nPost Processing: Converting logits to probabilities using a SoftMax layer:\n')
print(f'Logits converted to probabilities aka the model predictions: {predictions}\n')

"""
To get the labels corresponding to each position, we can inspect the id2label attribute of the model config
"""

print(f'Labels corresponding to each position:\n{model.config.id2label}\n')

# For Comparison
classifier = pipeline('sentiment-analysis')

print(classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!"
    ]
))

"""
Summary:

Model Used in the example: 'distilbert-base-uncased-finetuned-sst-2-english'

1.) Downloaded the tokenizer for this model
2.) Passed two raw input strings into our tokenizer, in order to be able to feed the input strings to the model

3.) Downloaded the base Transformer module
4.) Fed the tokenized input strings to the downloaded model
  - Model outputs 'hidden states' also known as 'features'
  - Hidden states are high dimensional vectors (these types of vectors usually have 3 dimensions)
  - Hidden states can be useful on their own, but they are usually inputs to another part of the model, known as the 'head'

5.) The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension.
  - Model outputs 'logits'
  - Logits are rae unnormalized scores outputted by the last layer of the model.

6.) Pass the logits through a softmax layer to convert them into probabilities.

"""
