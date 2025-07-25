from transformers import AutoModel, BertModel, AutoTokenizer
import torch

# The checkpoint name corresponds to a specific model architecture and weights
auto_model = AutoModel.from_pretrained('bert-base-cased') # BERT model with a basic architecture (l2 layers, 768 hidden size, and 12 attention layers) and cased inputs (meaning that the uppercase/lowercase distinction is important)

"""
The AutoModel class and its associates are actually simple wrappers designed to fetch the appropriate model architecture for a given checkpoint.
It's an 'auto' class meaning it will guess the appropriate model architecture for you and instantiate the correct model class.

However, if you know the type of model you want to use, you can use the class that defines it architecture directly:
"""

bert_model = BertModel.from_pretrained('bert-base-cased')

print(f'Auto Model: \n {auto_model}')

print(f'BERT Model: \n{bert_model}')

# Loading and saving
"""
Saving a model is as simple as saving a tokenizer.
Models have the same 'save_pretrained()' method, which saves the model's weights and architecture and architecture configuration.
"""

#bert_model.save_pretrained("/Users/dimitaratanassov/Hugging Face/chapter2/models") # Generates two files at the directory config.json and model.safetensors

"""
config.json file, contains all the necessary attributes needed to build the model architecture.
 This file also contains some metadata, such as where the checkpoint originated from and what Transformers version you were using when you last saved the checkpoint.

The model.safetensors file is known as the state dictionary; it contains all your model's weights.

These two files work together: 
the configuration file is needed to know about the model architecture, while the model weights are the parameters of the model.
"""

# Reusing a saved model, with the 'from_pretrained()' method again
#my_model = AutoModel.from_pretrained("/Users/dimitaratanassov/Hugging Face/chapter2/models")

# Need to login into huggingface with cmd: 'huggingface-cli login'
#my_model.push_to_hub("my-first-model") # This will upload the model files to the Hub, in a repository under your namespace named my-awesome-model.

# Anyone can now load your model with the 'from_pretrained()' method
#my_first_model = AutoModel.from_pretrained('DimitarAtanassov/my-first-model')

# Encoding Text
"""
Transformer models handle text by turning the inputs into numbers.
"""

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

encoded_input = bert_tokenizer("Hello, I'm a single sentence!")

print(f'Encoded input tokens: \n {encoded_input}')

"""
encoded_input is a dictionary with the following fields:
- input_ids: numerical representations of your tokens
- token_type_ids: these tell the model which part of the input is a sentence A and which is sentence B
- attention_mask: this indicates which tokens should be attended to and which should not
"""

# We can decode the input IDs to get back the original text:
decoded_input = bert_tokenizer.decode(encoded_input['input_ids'])
print(f'Decoded input: \n {decoded_input}')

"""
The tokenizer added special tokens when decoding the encoded_input: '[CLS]' and '[SEP]' -- these are required by the model (BERT).
    --Not all models need special tokens; they're utilized when a model was pretrained with them, 
    in which case the tokenizer needs to add them as that model expects these tokens.ß
"""

# You can encode multiple sentences at once, either by batching them togther or by passing a list:
encoded_input = bert_tokenizer("How are you?", "I'm fine, thank you!")
print(f"Encoded Input Tokens (2): \n {encoded_input}") # Note when passing multiple sentences, the tokenizer returns a list for each sentence for each dictionary value.

# We can also ask the tokenizer to return tensors directly from PyTorch:
encoded_input_tensors = bert_tokenizer("How are you?", "I'm fine, thank you!", return_tensors='pt')
print(f'Encoded Input Token Tensors (2): \n {encoded_input_tensors}')

"""
There's a problem: the two lists don't have the same length!
Arrays and tensors need to be rectangular, so we can't simply convert these lists to a PyTorch tensor (or NumPy array).
 * The Tokenizer provides an option for this: padding
"""

# Padding inputs

# If we ask the tokenizer to pad the inputs, it will make all the sentences the same length by adding a special padding token to the sentences that are shorter than the longest one:
padded_encoded_input_tensors = bert_tokenizer(
    ["How are?", "I'm fine, thank you!"], padding=True, return_tensors='pt'
) # Now we have rectangular tensors!
print(f'Encoded Padded Input Token Tensors (2): \n {padded_encoded_input_tensors}')

"""
Note that the padding tokens have been encoded into input IDs with ID 0, and they have an attention mask value of 0 as well.
 * This is becuase these padding tokens shouldn't be analyzed by the model: they're not part of the actual sentence.
"""

# Truncating inputs
"""
The tensors might get too big to be processed by the model.
 For instance, BERT was only pretrained with sequences up to 512 tokens, so it cannot process longer sequences.

If you have sequences longer than the model can handle, you'll need to truncate them with the 'truncation' parameter
"""
encoded_input = bert_tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(f'Truncated Encoded Input Tokens (1 very long sentence):\n {encoded_input['input_ids']}')

# By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need:
encoded_input = bert_tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors='pt'
)
print(f'Truncated and Padded Encoded Token Tensors: \n {encoded_input}')

# Adding special tokens
"""
Special tokens (or at least the concept of them) is particularly important to BERT and derived models.
These tokens are added to better represent the sentence boundaries, such as the beginning of a sentence ([CLS]) or separator between sentences ([SEP]).
"""

# A simple example
encoded_input = bert_tokenizer('How are you?')
print(f'Encoded Input Tokens with special tokens: \n {encoded_input}')
bert_tokenizer.decode(encoded_input["input_ids"])

"""
These special tokens are automatically added by the tokenizer.

Not all models need special tokens; they are primarily used when a model was pretrained with them, in which case the tokenizer will add them since the model expects them.
"""

# Why is all of this necessary? [An Example]
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
# tokenize the input
encoded_sequences = bert_tokenizer(sequences, padding=True) # Need to add padding to the second sentence so both input strings are the same length (We need to do this because Arrays and tensors need to be rectangular)

print(f'Encoded Sequence Tokens (2): \n {encoded_sequences.input_ids}')

model_inputs = torch.tensor(encoded_sequences.input_ids)

# Using the tensors as inputs to the model:
# Making use of the tensors with the model is extremely simple - we just call the model with the inputs:
model_output = bert_model(model_inputs)

print(f'Model Output for the Encoded Sequence Inputs (2): \n {model_output}')