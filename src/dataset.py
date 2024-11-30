import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from utils import canonicalize_text, read_file

## Sentiment treebank dataset
class SST5_Dataset(Dataset):
    def __init__(self, file_path):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.data = [
            (
                self.tokenizer(
                    canonicalize_text(row["sentence"]),
                    add_special_tokens=True,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
                int(row["label"]) if isinstance(row["label"], str) else row["label"],
            )
            for row in read_file(file_path)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        input_ids = X["input_ids"].squeeze(0)
        attention_mask = X["attention_mask"].squeeze(0)
        label = torch.tensor(y, dtype=torch.long)
        return input_ids, attention_mask, label


## Amazon reviews dataset
class Amazon_Dataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=512, is_training=True):
        self.data = []
        self.labels = []
        for text, label in zip(data, labels):
            # Canonicalize and tokenize the text
            canonicalized_text = canonicalize_text(text)
            tokens = tokenizer.tokenize(canonicalized_text)
            
            if is_training:
                # Perform chunking for training data
                chunks = [tokens[i:i + (max_length - 2)] for i in range(0, len(tokens), max_length - 2)]
                for chunk in chunks:
                    # Reconstruct text with special tokens
                    chunk_text = tokenizer.convert_tokens_to_string(chunk)
                    encoded_chunk = tokenizer(
                        chunk_text,
                        padding='max_length',
                        max_length=max_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    self.data.append(encoded_chunk)
                    self.labels.append(label)  # Assign the same label to all chunks of the same review
            else:
                # For non-training data, process without chunking
                encoded_text = tokenizer(
                    canonicalized_text,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                self.data.append(encoded_text)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids'].squeeze()
        attention_mask = self.data[idx]['attention_mask'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label
    
    
# Supervised contrastive learning dataset(NLI_format build from trainset)
class NLI_Dataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

    def pad_data(self, data):
        anchor = [x[0] for x in data]
        pos = [x[2] for x in data]
        neg = [x[4] for x in data]

        encoding_anchor = self.tokenizer(
            anchor, return_tensors="pt", padding=True, truncation=True
        )
        token_ids_anchor = encoding_anchor["input_ids"]
        attention_mask_anchor = encoding_anchor["attention_mask"]

        encoding_pos = self.tokenizer(
            pos, return_tensors="pt", padding=True, truncation=True
        )
        token_ids_pos = encoding_pos["input_ids"]
        attention_mask_pos = encoding_pos["attention_mask"]

        encoding_neg = self.tokenizer(
            neg, return_tensors="pt", padding=True, truncation=True
        )
        token_ids_neg = encoding_neg["input_ids"]
        attention_mask_neg = encoding_neg["attention_mask"]

        return (
            token_ids_anchor,
            attention_mask_anchor,
            token_ids_pos,
            attention_mask_pos,
            token_ids_neg,
            attention_mask_neg,
        )

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[1]))

        batches = []
        num_batches = int(np.ceil(len(all_data) / 32))

        for i in range(num_batches):
            start_idx = i * 32
            data = all_data[start_idx : start_idx + 32]

            (
                token_ids_anchor,
                attention_mask_anchor,
                token_ids_pos,
                attention_mask_pos,
                token_ids_neg,
                attention_mask_neg,
            ) = self.pad_data(data)

            batches.append(
                {
                    "token_ids_anchor": token_ids_anchor,
                    "attention_mask_anchor": attention_mask_anchor,
                    "token_ids_pos": token_ids_pos,
                    "attention_mask_pos": attention_mask_pos,
                    "token_ids_neg": token_ids_neg,
                    "attention_mask_neg": attention_mask_neg,
                }
            )

        return batches
    
    
# Unsupervised contrastive learning dataset(wiki1m dataset after sampling)    
class WikiDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                tokenized = tokenizer(
                    sentence,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                self.data.append(tokenized)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized = self.data[idx]
        input_ids = tokenized['input_ids'].squeeze(0) 
        attention_mask = tokenized['attention_mask'].squeeze(0)
        dummy_label = torch.tensor(-1, dtype=torch.long)
        return input_ids, attention_mask, dummy_label

