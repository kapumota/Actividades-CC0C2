## Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#### Componentes
class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # d_in: input dimension
        # d_out: output dimension (also used as the attention dimension)
        self.d_in = d_in
        self.d_out = d_out
        
        # Linear transformations for Query, Key, and Value
        self.Q = nn.Linear(d_in, d_out)
        self.K = nn.Linear(d_in, d_out)
        self.V = nn.Linear(d_in, d_out)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_in)
        
        # Transform input to Query, Key, and Value
        queries = self.Q(x)  # shape: (batch_size, seq_len, d_out)
        keys = self.K(x)     # shape: (batch_size, seq_len, d_out)
        values = self.V(x)   # shape: (batch_size, seq_len, d_out)
        
        # Compute attention scores
        # torch.bmm performs batch matrix multiplication
        scores = torch.bmm(queries, keys.transpose(1, 2))
        # shape: (batch_size, seq_len, seq_len)
        
        # Scale the scores
        scores = scores / (self.d_out ** 0.5)  # Apply scaling factor
        
        # Apply softmax to get attention weights
        attention = F.softmax(scores, dim=2)
        # shape: (batch_size, seq_len, seq_len)
        
        # Compute the weighted sum of values
        hidden_states = torch.bmm(attention, values)
        # shape: (batch_size, seq_len, d_out)
        
        return hidden_states
    

class BasicMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size  # Total hidden size
        self.num_heads = num_heads      # Number of attention heads
        
        # Linear layer to combine outputs from all heads
        self.out = nn.Linear(hidden_size, hidden_size)
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            Attention(hidden_size, hidden_size // num_heads) 
            for _ in range(num_heads)
        ])
        # Each head operates on a slice of the hidden state
        # hidden_size // num_heads is the size of each head's output
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        
        # Apply each attention head to the input
        outputs = [head(x) for head in self.heads]
        # Each output shape: (batch_size, seq_len, hidden_size // num_heads)
        
        # Concatenate the outputs from all heads
        outputs = torch.cat(outputs, dim=2)
        # shape after concatenation: (batch_size, seq_len, hidden_size)
        
        # Apply the output linear transformation
        hidden_states = self.out(outputs)
        # shape: (batch_size, seq_len, hidden_size)
        
        return hidden_states


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Single linear layer for Q, K, V projections
        self.qkv_linear = nn.Linear(hidden_size, hidden_size * 3)
        # Output projection
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        
        # Project input to Q, K, V
        # (batch_size, seq_length, hidden_size * 3)
        qkv = self.qkv_linear(x)
        
        # Reshape and transpose for multi-head attention
        # (batch_size, seq_length, num_heads, head_dim * 3)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        # (batch_size, num_heads, seq_length, head_dim * 3)
        qkv = qkv.transpose(1, 2)
        
        # Split into Q, K, V
        # Each of shape (batch_size, num_heads, seq_length, head_dim)
        queries, keys, values = qkv.chunk(3, dim=-1)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_length, seq_length)
        scores = torch.matmul(queries, keys.transpose(2, 3))
        
        # Scale scores
        # (batch_size, num_heads, seq_length, seq_length)
        scores = scores / (self.head_dim ** 0.5)
        
        # Apply softmax to get attention weights
        # (batch_size, num_heads, seq_length, seq_length)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_length, head_dim)
        context = torch.matmul(attention, values)
        
        # Transpose and reshape to combine heads
        # (batch_size, seq_length, num_heads, head_dim)
        context = context.transpose(1, 2)
        # (batch_size, seq_length, hidden_size)
        context = context.reshape(batch_size, seq_length, hidden_size)
        
        # Apply output projection
        # (batch_size, seq_length, hidden_size)
        output = self.out(context)
        
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model):
        super().__init__()
        
        # Create a matrix of shape (context_size, d_model) to store the positional encodings
        encoding = torch.zeros(context_size, d_model)
        
        # Create a tensor of positions from 0 to context_size - 1
        # Shape: (context_size, 1)
        pos = torch.arange(0, context_size).unsqueeze(dim=1)
        
        # Create a tensor of even indices from 0 to d_model - 2
        # Shape: (d_model / 2)
        dim = torch.arange(0, d_model, 2)
        
        # Compute the arguments for the sine and cosine functions
        # Shape: (context_size, d_model / 2)
        arg = pos / (10000 ** (dim / d_model))
        
        # Compute sine values for even indices
        encoding[:, 0::2] = torch.sin(arg)
        
        # Compute cosine values for odd indices
        encoding[:, 1::2] = torch.cos(arg)

        self.register_buffer('encoding', encoding)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Return the positional encoding for the given sequence length
        return self.encoding[:, :seq_len, :]
    

class EfficientPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a tensor of positions from 0 to max_len - 1
        # Shape: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        
        # Compute the division terms for the encoding
        # This is an optimization of the original formula
        # Shape: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Initialize the positional encoding tensor
        # Shape: (max_len, 1, d_model)
        pe = torch.zeros(max_len, 1, d_model)
        
        # Compute sine values for even indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        # Compute cosine values for odd indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer
        # This means it won't be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Return the positional encoding for the given sequence length
        return self.pe[:, :seq_len, :]
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # First linear transformation
        # Increases dimensionality from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation
        # Decreases dimensionality back from d_ff to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        # Apply first linear transformation
        x = self.linear1(x)
        
        # Apply ReLU activation
        x = self.relu(x)
        
        # Apply second linear transformation
        x = self.linear2(x)
        
        # Output shape: (batch_size, seq_len, d_model)


        return x


### Transformers.py 


import torch
import torch.nn as nn
import torch.nn.functional as F

from components import (
    PositionalEncoding,
    PositionwiseFeedForward
)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Multi-head Self-Attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        
        # Self-Attention
        # Note: PyTorch's MultiheadAttention expects input in shape (seq_len, batch_size, d_model)
        hidden_states, _ = self.self_attn(query=x, key=x, value=x)
        
        # Add & Norm (Residual connection and Layer Normalization)
        x = self.norm1(x + hidden_states)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        
        # Add & Norm (Residual connection and Layer Normalization)
        x = self.norm2(x + ff_output)
        
        # Output shape: (seq_len, batch_size, d_model)
        return x
    

class Encoder(nn.Module):
    def __init__(self, input_size, context_size, 
                 d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(input_size, d_model)
        
        # Positional encoding to add position information
        self.pos_embedding = PositionalEncoding(context_size, d_model)

        # Stack of Encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # Apply embedding and add positional encoding
        x = self.embedding(x)
        x += self.pos_embedding(x)
        # x shape after embedding: (batch_size, seq_len, d_model)
        
        # Pass through each encoder block
        for block in self.blocks:
            x = block(x)
        
        # Output shape: (batch_size, seq_len, d_model)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Self-Attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Cross-Attention layer
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output):
        # x shape: (seq_len, batch_size, d_model)
        # enc_output shape: (enc_seq_len, batch_size, d_model)
        
        # Self-Attention
        hidden_states, _ = self.self_attn(x, x, x)
        x = self.norm1(x + hidden_states)
        
        # Cross-Attention
        hidden_states, _ = self.cross_attn(query=x, key=enc_output, value=enc_output)
        x = self.norm2(x + hidden_states)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        # Output shape: (seq_len, batch_size, d_model)
        return x
    

class Decoder(nn.Module):
    def __init__(self, output_size, context_size, d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(output_size, d_model)
        
        # Positional encoding to add position information
        self.pos_embedding = PositionalEncoding(context_size, d_model)
        
        # Stack of Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(n_blocks)
        ])

        # Output linear layer
        self.out = nn.Linear(d_model, output_size)

    def forward(self, x, enc_output):
        # x shape: (batch_size, seq_len)
        # enc_output shape: (enc_seq_len, batch_size, d_model)
        
        # Apply embedding and add positional encoding
        x = self.embedding(x)
        x += self.pos_embedding(x)
        # x shape after embedding: (batch_size, seq_len, d_model)
        
        # Pass through each decoder block
        for block in self.blocks:
            x = block(x, enc_output)
        
        # Project to output size
        output = self.out(x)
        # output shape: (batch_size, seq_len, output_size)
        
        return output
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, d_ff, num_heads, n_blocks):
        super().__init__()
        
        # Encoder component
        self.encoder = Encoder(
            vocab_size,     # Size of the input vocabulary
            context_size,   # Maximum sequence length
            d_model,        # Dimensionality of the model
            d_ff,           # Dimensionality of the feedforward network
            num_heads,      # Number of attention heads
            n_blocks        # Number of encoder blocks
        )
        
        # Decoder component
        self.decoder = Decoder(
            vocab_size,     # Size of the output vocabulary (same as input in this case)
            context_size,   # Maximum sequence length
            d_model,        # Dimensionality of the model
            d_ff,           # Dimensionality of the feedforward network
            num_heads,      # Number of attention heads
            n_blocks        # Number of decoder blocks
        )

    def forward(self, input_encoder, input_decoder):
        # input_encoder shape: (batch_size, enc_seq_len)
        # input_decoder shape: (batch_size, dec_seq_len)
        
        # Pass input through the encoder
        enc_output = self.encoder(input_encoder)
        # enc_output shape: (batch_size, enc_seq_len, d_model)
        
        # Pass encoder output and decoder input through the decoder
        output = self.decoder(input_decoder, enc_output)
        # output shape: (batch_size, dec_seq_len, vocab_size)
        
        return output



### Testing.py 
import torch
from transformer import Transformer

# Define special tokens
SOS_token = 0  # Start of sequence token
EOS_token = 1  # End of sequence token
PAD_token = 2  # Padding token

# Initialize vocabulary with special tokens
index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS',
    PAD_token: 'PAD'
}

# Sample sentence to build vocabulary
words = "How are you doing ? I am good and you ?"
words_list = set(words.lower().split(' '))
for word in words_list:
    index2words[len(index2words)] = word
    
# Create reverse mapping: word to index
words2index = {w: i for i, w in index2words.items()}

def convert2tensors(sentence, max_len):
    """Convert a sentence to a padded tensor of word indices."""
    words_list = sentence.lower().split(' ')
    padding = ['PAD'] * (max_len - len(words_list))
    words_list.extend(padding)
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

# Set model hyperparameters
D_MODEL = 10
VOCAB_SIZE = len(words2index)
N_BLOCKS = 10
D_FF = 20
CONTEXT_SIZE = 100
NUM_HEADS = 2

# Initialize the Transformer model
transformer = Transformer(
    vocab_size=VOCAB_SIZE, 
    context_size=CONTEXT_SIZE, 
    d_model=D_MODEL, 
    d_ff=D_FF, 
    num_heads=NUM_HEADS, 
    n_blocks=N_BLOCKS
)

# Prepare input sentences
input_sentence = "How are you doing ?"
output_sentence = "I am good and"

# Convert sentences to tensors
input_encoder = convert2tensors(input_sentence, CONTEXT_SIZE)
input_decoder = convert2tensors(output_sentence, CONTEXT_SIZE)

# Run the model
output = transformer(input_encoder, input_decoder)

# Get the most likely next word
_, indexes = output.squeeze().topk(1)
predicted_word = index2words[indexes[3].item()]
print(f"Predicted next word: {predicted_word}")
# > 'are' (for example)


##Training LLM


Pretraining

%pip install torch transformers[torch] datasets ipywidgets trl

from datasets import load_dataset

wiki_data = load_dataset(
    "wikimedia/wikipedia",   
    "20231101.en", 
    split="train[:1000]"
)

print(wiki_data['text'][0][:1000])
wiki_data = wiki_data.train_test_split(test_size=0.2)

wiki_data

from transformers import AutoTokenizer

base_model_id = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.special_tokens_map

outputs = tokenizer(
    wiki_data['train']['text'][0:10],
)

max_length = 512

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=max_length, 
        padding='max_length', # longuest 
        return_tensors="pt", 
        add_special_tokens=True
    )

tokenized_datasets = wiki_data.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['id', 'url', 'title', 'text']
)

tokenizer.padding_side


tokenizer.pad_token_id


from transformers import MistralForCausalLM, MistralConfig
config = MistralConfig()

config

config = MistralConfig(
    hidden_size=768,
    sliding_window=768,
    intermediate_size=3072,
    max_position_embeddings=max_length,
    num_attention_heads=16,  
    num_hidden_layers=4,
)

model = MistralForCausalLM(config)

model

model_size = sum(t.numel() for t in model.parameters())
model_size

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([
    tokenized_datasets["train"][i] for i in range(10)
])


out['input_ids'][0]

out['labels'][0]

from huggingface_hub import notebook_login

notebook_login()



from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="mistral-pretraining",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    push_to_hub=True,
    report_to="none", 
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

model.device

trainer.push_to_hub()

from transformers import pipeline

model_id = "damienbenveniste/mistral-pretraining"
pipe = pipeline("text-generation", model=model_id)
txt = "How are you?"
pipe(txt, num_return_sequences=1)

txt = "generated_text"
pipe(txt)


Supervised Learning Fine-tuning

dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

dataset['instruction'][0]
out = dataset[0]['text']
print(out)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "damienbenveniste/mistral-pretraining"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.config.pad_token_id

from trl import DataCollatorForCompletionOnlyLM

response_template = "\n### Response:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]

data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

print(dataset['text'][1])


tokenized = tokenizer(
    dataset['text'][1], 
)

data_collator([tokenized])



from trl import SFTTrainer, SFTConfig

args = SFTConfig(
    output_dir="mistral-supervised",
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=1,
    push_to_hub=True,
    report_to="none", 
)

# args = TrainingArguments(
#     output_dir="mistral-supervised",
#     num_train_epochs=1,
#     push_to_hub=True,
#     report_to="none", 
# )

trainer = SFTTrainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=data_collator,
    # dataset_text_field='text',  
    # max_seq_length=512, 
)

trainer.train()

trainer.push_to_hub()


RLHF

dataset = load_dataset("Anthropic/hh-rlhf", split='train[:1000]')
dataset['chosen'][0]
print(dataset[1]['rejected'])

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

tokenized_data = dataset.map(
    preprocess_function,
    batched=True,
)


from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = 'damienbenveniste/mistral-supervised'

reward_model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

reward_model.config.pad_token_id = tokenizer.pad_token_id

from trl import RewardTrainer, RewardConfig

reward_config = RewardConfig(
    output_dir="mistral-reward",
    num_train_epochs=1,
    push_to_hub=True,
    report_to="none",
)

trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=tokenized_data,
)

trainer.train()

trainer.push_to_hub()


## PPO Training

dataset = load_dataset("tatsu-lab/alpaca", split="train[-1000:]")

dataset['text'][0]

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

model_id = 'damienbenveniste/mistral-supervised'

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

ppo_model

print(dataset['text'][1].split('### Response')[0].strip())

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(
        sample["text"].split('### Response')[0].strip(), 
    )
    sample["prompt"] = sample["text"].split('### Response')[0].strip()
    return sample

tokenized_dataset = dataset.map(tokenize, batched=False)
tokenized_dataset.set_format(type="torch")

tokenized_dataset[0]

def collator(data):    
    return dict((key, [d[key] for d in data]) for key in data[0])

tokenized_dataset[0]


collated = collator(tokenized_dataset)

from trl import PPOConfig, PPOTrainer
from transformers import pipeline

ppo_config = PPOConfig(
    remove_unused_columns=False,
    mini_batch_size=2,
    batch_size=2,
)

ppo_trainer = PPOTrainer(
    model=ppo_model,
    config=ppo_config,
    dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

reward_pipeline = pipeline(model='damienbenveniste/mistral-reward')

batch = next(iter(ppo_trainer.dataloader))
batch['input_ids']
query_tensors = batch['input_ids']

response_tensors = ppo_trainer.generate(
    query_tensors, 
    pad_token_id=tokenizer.eos_token_id,
    return_prompt=False,
    min_length=-1,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    max_new_tokens=10
)

response_tensors


batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

batch['response']

['species, they."). As theune coverage of',
 'Stewart and support of hero and C no brightonial']

def get_text(instruction, response):
    return 'Human: {} \n\n Assistant: {}'.format(instruction, response)

texts = [get_text(q, r) for q, r in zip(batch["instruction"], batch["response"])]

texts

outputs = reward_pipeline(texts)

outputs

import torch
rewards = [torch.tensor(output["score"]) for output in outputs]

rewards

[tensor(0.5694), tensor(0.5224)]

ppo_trainer.step(
    query_tensors, 
    response_tensors, 
    rewards
)  

epochs = 1
for epoch in range(epochs):
    for batch in ppo_trainer.dataloader: 
        query_tensors = batch["input_ids"]    
        
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            pad_token_id=tokenizer.eos_token_id,
            return_prompt=False,
            min_length=-1,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            max_new_tokens=10
        )
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [get_text(q, r) for q, r in zip(batch["instruction"], batch["response"])]
        outputs = reward_pipeline(texts)
        rewards = [torch.tensor(output["score"]) for output in outputs]
    
        #### Run PPO step
        ppo_trainer.step(
            query_tensors, 
            response_tensors, 
            rewards
        ) 
        # break    

ppo_trainer.push_to_hub('mistral-ppo')










#Finetuning LoRA-QLora
%pip install -U peft transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

tokenizer.special_tokens_map
from datasets import load_dataset

spanish_data = load_dataset('andreamorgar/spanish_poetry')
french_data = load_dataset('Abirate/french_book_reviews')

spanish_data['train']['content']
french_data['train']['reader_review']

max_length = 128

def preprocess_spanish(examples):
    return tokenizer(
        [x for x in examples['content'] if x], 
        max_length=max_length,
        truncation=True, 
        padding='max_length'
    )

def preprocess_french(examples):
    return tokenizer(
        [x for x in examples['reader_review'] if x], 
        max_length=max_length,
        truncation=True, 
        padding='max_length'
    )

tokenized_spanish = spanish_data.map(
    preprocess_spanish,
    batched=True,
    remove_columns=spanish_data['train'].column_names,
)

tokenized_french = french_data.map(
    preprocess_french,
    batched=True,
    remove_columns=french_data['train'].column_names,
)


tokenized_french

from peft import LoraConfig

lora_config = LoraConfig(
    r=64,
    task_type="CAUSAL_LM",
    # target_modules={'q_proj', 'v_proj', 'embed_tokens'}
)

print(lora_config)

model = AutoModelForCausalLM.from_pretrained(model_id)
model
model.add_adapter(lora_config, adapter_name='spanish_adapter')
model
model.add_adapter(lora_config, adapter_name='french_adapter')
model
model.active_adapters()
model.set_adapter('spanish_adapter')
model.active_adapters()
model = AutoModelForCausalLM.from_pretrained(model_id)
model
from peft import get_peft_model

peft_model = get_peft_model(
    model, 
    lora_config, 
    adapter_name='spanish_adapter'
)

peft_model
peft_model.get_base_model()
import peft

peft.__version__

peft_model.add_adapter(
    adapter_name='french_adapter', 
    peft_config=lora_config
)

peft_model

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./result_training",
    learning_rate=2e-5,
    weight_decay=0.01,
)

peft_model.set_adapter('spanish_adapter')

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_spanish['train'],
    data_collator=data_collator,
)

trainer.train()


base_model = peft_model.get_base_model()

def generate_text(prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(output[0]) 

base_model.to('cpu')
generate_text('Como estas?', base_model)



peft_model.to('cpu')
peft_model.set_adapter('spanish_adapter')
generate_text('Como estas?', peft_model)

training_args = TrainingArguments(
    output_dir="./result_training",
    learning_rate=2e-5,
    weight_decay=0.01,
)

peft_model.to('mps')
peft_model.set_adapter('french_adapter')

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_french['train'],
    data_collator=data_collator,
)

trainer.train()

base_model.to('cpu')
generate_text('Comment ca va?', base_model)

peft_model.set_adapter('french_adapter')
generate_text('Comment ca va?', peft_model)

peft_model.save_pretrained('peft_adapters')


from peft import PeftModelForCausalLM

model_spanish = PeftModelForCausalLM.from_pretrained(
    model,
    'peft_adapters/spanish_adapter'
)

peft_model.add_weighted_adapter(
    ['spanish_adapter', 'french_adapter'], 
    [0.5, 0.5], 
    adapter_name='new_adapter')

peft_model

inputs = tokenizer(
    [
        "Hello",
        "Bonjour",
        "Hola",
    ],
    return_tensors="pt",
    padding=True,
)

adapter_names = [
    "__base__", 
    "french_adapter",
    "spanish_adapter",
]

peft_model.eval()

output = peft_model.generate(
    **inputs, 
    adapter_names=adapter_names, 
    max_new_tokens=20
)

tokenizer.decode(output[0]) 
tokenizer.decode(output[1]) 
tokenizer.decode(output[2]) 

%pip install -U bitsandbytes


import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=config,
)

model = prepare_model_for_kbit_training(model)

from peft import get_peft_model

peft_model = get_peft_model(
    model, 
    lora_config, 
    adapter_name='spanish_adapter'
)

%pip install autoawq



%pip install quanto

from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from peft import prepare_model_for_kbit_training

quantization_config = QuantoConfig(weights="int8")
model_id = "facebook/opt-350m"
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
)
# quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=quantization_config)

quantized_model = prepare_model_for_kbit_training(quantized_model)

QuantoConfig().to_dict()



###deploying-LLM-basic

from openai import OpenAI

openai_api_base = "http://ec2-18-144-41-161.us-west-1.compute.amazonaws.com:8000/v1"

client = OpenAI(
    api_key='none',
    base_url=openai_api_base,
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "What is the history of machine learning?"}
    ],
    stream=True,
    temperature=0.7,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


pip install --upgrade vllm Jinja2 jsonschema
export PATH="$PATH:$HOME/.local/bin"
export HF_TOKEN='YOUR TOKEN'
vllm serve meta-llama/Llama-2-7b-chat-hf --dtype half --max-model-len 200
