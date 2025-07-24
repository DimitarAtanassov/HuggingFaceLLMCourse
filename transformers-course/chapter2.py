from transformers import pipeline, AutoTokenizer, AutoModel

# classifier = pipeline('sentiment-analysis')

# print(classifier(
#     [
#         "I've been waiting for a HuggingFace course my whole life.",
#         "I hate this so much!"
#     ]
# ))

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

print(f'\nTokenizer Created: {tokenizer}\n')

"""
Once we have our tokenizer, we can directly pass our sentences to it and we'll get back a dictionary that's ready to feed to our model.
The only thing left to do is to convert the list of input IDs to tensors

Transformer models only accept tensors as input.
- If this is your first time hearing about tensors, you can think of them as NumPy arrays instead.
- A NumPy array can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions.
"""

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
print(f'Tokenized Input Strings: {model_inputs}\n')

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
