import os
import random
import shutil
import time
from typing import Generator
from typing import List
from typing import Tuple
import nltk
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def create_trimmed_data_loader(dataset, batch_size, tokenizer, max_length=512):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trimmed_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, max_length=max_length, return_special_tokens_mask=True), batched=True, remove_columns=['text'])
    trimmed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_loader = torch.utils.data.DataLoader(trimmed_dataset, batch_size=batch_size, collate_fn=data_collator)
    return data_loader


class DatasetPreparationPipeline:
    def __init__(self, data_dir, config, tokenizer):
        self.data_dir = data_dir
        self.config = config
        self.tokenizer = tokenizer

    def calculate_target_dataset_size(self) -> int:
        self.config.print_parameter_count()
        target_tokens = self.config.total_params * 20
        return target_tokens

    def tokenize_until_target_size(self, target_tokens: int, sample_size: float = 0.02) -> Tuple[int, list]:
        file_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        random.shuffle(file_paths)

        tokenized_files = []
        total_tokens = 0

        while total_tokens < target_tokens:
            sample_files = file_paths[:int(len(file_paths) * sample_size)]
            sample_tokens = []
            for file_path in sample_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                sample_tokens.extend(self.tokenizer.encode(text))
                tokenized_files.append(file_path)

            total_tokens += len(sample_tokens)
            file_paths = file_paths[int(len(file_paths) * sample_size):]

        print(f"Tokenized {len(tokenized_files)} files ({(len(tokenized_files) / len(os.listdir(self.data_dir))) * 100:.2f}% of dataset).")
        print(f"Total tokens in dataset: {total_tokens:,}")

        return total_tokens, tokenized_files

    def create_dataset_directory(self, tokenized_files: list):
        dataset_dir = os.path.join(self.data_dir, f"dataset__{self.config.num_hidden_layers}_{self.config.hidden_size}_{self.config.num_attention_heads}_{self.config.max_position_embeddings}")
        os.makedirs(dataset_dir, exist_ok=True)

        for file_path in tokenized_files:
            shutil.copy(file_path, dataset_dir)

        print(f"Created dataset directory: {dataset_dir}")

    def tokenize_dataset(self, dataset_dir, tokenizer, config):
        rows = []
        document_id = 0
        total_paragraphs = 0
        paragraph_counter = 0

        file_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.txt')]

        for file_path in tqdm(file_paths, desc="Tokenizing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            paragraphs = self.create_paragraphs(text, config.max_position_embeddings)
            num_paragraphs = len(paragraphs)
            total_paragraphs += num_paragraphs
            print(f"Number of paragraphs in file {file_path}: {num_paragraphs}")

            for i, paragraph in enumerate(paragraphs):
                encoding = tokenizer.encode_plus(
                    paragraph,
                    max_length=config.max_position_embeddings,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=False,
                    return_tensors="pt"
                )

                row = {
                    'document_id': document_id,
                    'text': paragraph,
                    'tokens': tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist()),
                    'token_ids': encoding['input_ids'].squeeze().tolist()
                }

                rows.append(row)
                paragraph_counter += 1
                print(f"Paragraph counter: {paragraph_counter}")

            document_id += 1

        print(f"Total paragraphs: {total_paragraphs}")
        return rows

    def create_paragraphs(self, text: str, max_length: int) -> List[str]:
        # Tokenize the text into paragraphs using NLTK
        paragraphs = text.split('\n\n')

        result_paragraphs = []
        current_paragraph = ""

        for paragraph in paragraphs:
            if len(current_paragraph) + len(paragraph) + 1 <= max_length:
                current_paragraph += " " + paragraph if current_paragraph else paragraph
            else:
                if current_paragraph:
                    result_paragraphs.append(current_paragraph)
                current_paragraph = paragraph

        if current_paragraph:
            result_paragraphs.append(current_paragraph)

        return result_paragraphs



class AtomicRobertaConfig:
    """
    Configuration class for the AtomicRoBERTa model.

    AtomicRoBERTa is a compact and lightweight variant of the RoBERTa model, designed for efficient deployment on
    resource-constrained devices or for use cases where a smaller model size is preferred. It strikes a balance
    between model performance and parameter count, offering a more efficient alternative to larger language models.


    INSTRUCTIONS and
    TODO:
        Yes, that's correct. The memory management technique used here is based on the Hugging Face Transformers library and its data loading utilities.
        Here's how it works:
        Dataset Loading: First, you need to load your dataset into a Hugging Face Dataset object. This can be done using the datasets library or by creating a custom Dataset object from your data files.
        Tokenization and Truncation: In the create_trimmed_data_loader function, we're using the tokenizer to tokenize and truncate the text data to a maximum length (max_length=512 by default). This is done using the map method of the dataset, which applies the tokenization function to each example in the dataset.
        Dataset Formatting: After tokenization and truncation, the dataset is formatted to PyTorch tensors using trimmed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask']). This ensures that the input data is in the correct format for PyTorch models.
        Data Collator: The DataCollatorForLanguageModeling is used to handle the batching and padding of the input sequences. It ensures that the input sequences within a batch are padded to the same length, and it creates the appropriate attention masks for the model.
        Data Loader: Finally, a PyTorch DataLoader is created from the trimmed and formatted dataset. The DataLoader handles the batching of the data during training or inference. The collate_fn argument is set to the data_collator object, which performs the necessary padding and batching operations.
        Training or Inference: During training or inference, you can iterate over the data_loader to get batches of input data. The model will process these batches one by one, and the memory management is handled automatically by PyTorch and the Hugging Face Transformers library.
        To summarize, this approach uses the Hugging Face Transformers library to load and preprocess the data, truncate sequences to a maximum length, pad and batch the input sequences, and create a data loader that can be used to feed data to the model in a memory-efficient manner.
        As for batching, yes, the data loader batches the input sequences, but it doesn't necessarily batch one paragraph at a time. The batch size is determined by the batch_size argument passed to the create_trimmed_data_loader function. Each batch will contain batch_size number of input sequences, which could be paragraphs, sentences, or any other text input, depending on your dataset.



    """

    def __init__(self,
                 vocab_size: int = 50265,
                 max_position_embeddings: int = 512,
                 num_attention_heads: int = 6,
                 num_hidden_layers: int = 4,
                 hidden_size: int = 256,
                 intermediate_size: int = 512,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 pad_token_id: int = 1,
                 bos_token_id: int = 0,
                 eos_token_id: int = 2,
                 **kwargs):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def print_parameter_count(self):
        """
        Prints the approximate total number of parameters in the AtomicRoBERTa model.

        The parameter count is calculated as follows:
        - Embedding layer: (vocab_size * hidden_size) + (max_position_embeddings * hidden_size) + (2 * hidden_size)
        - Encoder layers: num_hidden_layers * (4 * hidden_size^2 + 2 * hidden_size * intermediate_size)
        """

        embedding_params = (self.vocab_size * self.hidden_size) + (self.max_position_embeddings * self.hidden_size) + (2 * self.hidden_size)
        encoder_params = self.num_hidden_layers * (4 * self.hidden_size**2 + 2 * self.hidden_size * self.intermediate_size)
        self.total_params = embedding_params + encoder_params

        print(f"Approximate total number of parameters in AtomicRoBERTa: {self.total_params:,}")


class AtomicRobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class AtomicRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob, batch_first=True)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = nn.functional.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        layer_output = self.dropout(layer_output)
        return layer_output


class AtomicRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = AtomicRobertaEmbeddings(config)
        self.encoder = nn.ModuleList([AtomicRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_indices=None):
        embeddings = self.embeddings(input_ids, token_type_ids)
        sequence_output = embeddings
        for layer in self.encoder:
            sequence_output = layer(sequence_output, attention_mask)

        if masked_indices is not None:
            # Only compute the loss for masked tokens
            masked_sequence_output = sequence_output[masked_indices]
            masked_logits = self.mlm_head(masked_sequence_output)
            return masked_logits
        else:
            # Return the sequence output for inference
            return sequence_output


def estimate_dataset_size(data_dir, sample_size=0.02):
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # Get all file paths
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    # Tokenize a sample of files
    sample_files = file_paths[:int(len(file_paths) * sample_size)]
    sample_tokens = []
    for file_path in sample_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sample_tokens.extend(tokenizer.encode(text))

    # Estimate total tokens
    total_tokens = len(sample_tokens) / sample_size

    # Print estimates
    print(f"Tokenized {len(sample_files)} files ({sample_size * 100}% of dataset).")
    print(f"Estimated total tokens in dataset: {int(total_tokens):,}")

    # Calculate target dataset size based on scaling laws
    config = AtomicRobertaConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, max_position_embeddings=512)
    config.print_parameter_count()
    target_tokens = config.total_params * 20
    print(f"Target dataset size (20x parameters): {int(target_tokens):,} tokens")

    return total_tokens, target_tokens


# Usage example

config = AtomicRobertaConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, max_position_embeddings=512)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
data_dir = "C:\\Users\\doren\\AppData\\Roaming\\Gantrithor\\data\\datasets\\ATOMIC_ROBERTA_TRAINING_DATA"

dataset_dir = os.path.join(data_dir, f"dataset__{config.num_hidden_layers}_{config.hidden_size}_{config.num_attention_heads}_{config.max_position_embeddings}")
saved_dataset_dir = f"{dataset_dir}_saved_dataset"

# if os.path.exists(dataset_dir):
#     print(f"Dataset directory already exists: {dataset_dir}")
#     pipeline = DatasetPreparationPipeline(data_dir, config, tokenizer)
#     rows = pipeline.tokenize_dataset(dataset_dir, tokenizer, config)
#     tokenized_dataset = Dataset.from_pandas(pd.DataFrame(rows))
#     tokenized_dataset.save_to_disk(dataset_dir)
# else:
#     print("Dataset directory does not exist. Please create it first.")
#     pipeline = DatasetPreparationPipeline(data_dir, config, tokenizer)
#     target_tokens = pipeline.calculate_target_dataset_size()
#     total_tokens, tokenized_files = pipeline.tokenize_until_target_size(target_tokens)
#     pipeline.create_dataset_directory(tokenized_files)
#     if not os.path.exists(saved_dataset_dir):
#         rows = pipeline.tokenize_dataset(dataset_dir, tokenizer, config)
#         tokenized_dataset = Dataset.from_pandas(pd.DataFrame(rows))
#         tokenized_dataset.save_to_disk(saved_dataset_dir)
#     else:
#         pass

# Load the tokenized dataset
dataset_dir = os.path.join(data_dir, f"dataset__{config.num_hidden_layers}_{config.hidden_size}_{config.num_attention_heads}_{config.max_position_embeddings}")
tokenized_dataset = Dataset.load_from_disk(dataset_dir)

def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    mask_token_id = tokenizer.mask_token_id
    input_ids = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    input_ids[masked_indices] = mask_token_id
    return input_ids, masked_indices

data_loader = DataLoader(
    tokenized_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda batch: {
        'input_ids': torch.stack([torch.tensor(example['token_ids']) for example in batch]),
        'attention_mask': torch.ones_like(torch.stack([torch.tensor(example['token_ids']) for example in batch]), dtype=torch.bool),
        'masked_input_ids': mask_tokens(torch.stack([torch.tensor(example['token_ids']) for example in batch]), tokenizer)[0],
        'masked_indices': mask_tokens(torch.stack([torch.tensor(example['token_ids']) for example in batch]), tokenizer)[1]
    }
)

# Initialize the model
# Initialize the model
model = AtomicRobertaModel(config)

# Set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 2
batch_count = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        masked_input_ids = batch['masked_input_ids']
        masked_indices = batch['masked_indices']

        batch_size = masked_input_ids.size(0)
        attention_mask = attention_mask.view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(batch_size, config.num_attention_heads, masked_input_ids.size(1), -1)
        attention_mask = attention_mask.reshape(batch_size * config.num_attention_heads, masked_input_ids.size(1), -1)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0

        optimizer.zero_grad()
        masked_logits = model(masked_input_ids, attention_mask=attention_mask, masked_indices=masked_indices)
        loss = criterion(masked_logits.view(-1, config.vocab_size), input_ids[masked_indices].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch_count += 1
        if batch_count % 2048 == 0:
            print("batch counter: ", batch_count)

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

try:
    # Save the trained model
    torch.save(model.state_dict(), 'atomic_roberta_model.pt')
except Exception as e:
    print("error: ", e)
# Save the entire model
model_dir = r"C:\Users\doren\PycharmProjects\GANTRITHOR_FINAL_2024\TESTCODE\ATOMIC_ROBERTA"
torch.save(model, model_dir)

print("Creating data loader...")
# data_loader = create_trimmed_data_loader(tokenized_dataset, batch_size=8, tokenizer=tokenizer, max_length=config.max_position_embeddings)

print("finished")

# time.sleep(10000)

# Create the data loader



