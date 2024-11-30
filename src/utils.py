import random
from nltk.tree import Tree
import re
import unicodedata
from tqdm import tqdm
from transformers import BertTokenizer
import yaml
import os

## Read Penn Treebank tree
def read_ptb_tree(tree_string):
    return Tree.fromstring(tree_string)

## Extract sentence and label from Penn Treebank tree
def extract_sentence_and_label(tree):
    label = tree.label()

    words = tree.leaves()
    sentence = " ".join(words)

    return sentence, label

## Read data from sst file
def read_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            tree = read_ptb_tree(line.strip())
            sentence, label = extract_sentence_and_label(tree)
            data.append({"sentence": sentence, "label": label})
    return data


## Group data by sentiment level for supervised contrastive learning
def group_data_by_level(data):
    data_by_level = {}

    for item in data:
        label = int(item["label"])
        sentence = item["sentence"]

        if label not in data_by_level:
            data_by_level[label] = []

        data_by_level[label].append(sentence)

    return data_by_level

## Create large data pairs for supervised contrastive learning
def create_large_data_pairs(data_by_level, target_size):
    pairs = []
    levels = list(data_by_level.keys())

    while len(pairs) < target_size:
        for level in levels:
            level_data = data_by_level[level]
            if len(level_data) < 2:
                continue

            sen0, sent1 = random.sample(level_data, 2)

            if level == 0:
                hard_neg_level = 1
            elif level == 4:
                hard_neg_level = 3
            else:
                hard_neg_level = random.choice([level - 1, level + 1])

            hard_neg = random.choice(data_by_level[hard_neg_level])

            pairs.append((sen0, sent1, hard_neg))

            if len(pairs) >= target_size:
                break

    return pairs

## Canonicalize text
def canonicalize_text(text):
    text = re.sub(r"[\d\W_]+", " ", text)

    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )

    text = text.lower()

    text = text.strip()

    return text

## Create triples for supervised contrastive learning
def create_triples(pairs):
    triples = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for sen0, sent1, hard_neg in tqdm(pairs):

        anchor = canonicalize_text(sen0)
        pos = canonicalize_text(sent1)
        neg = canonicalize_text(hard_neg)

        anchor_tokens = tokenizer.tokenize("[CLS] " + anchor + " [SEP]")
        pos_tokens = tokenizer.tokenize("[CLS] " + pos + " [SEP]")
        neg_tokens = tokenizer.tokenize("[CLS] " + neg + " [SEP]")

        triples.append((anchor, anchor_tokens, pos, pos_tokens, neg, neg_tokens))

    return triples

## Read Amazon reviews
def read_amazon_reviews(data_path):
    data = {"train": {}, "dev": {}, "test": {}}

    # Duyệt qua các thư mục chính: train, dev, test
    for split in data.keys():
        split_path = os.path.join(data_path, split)
        sentiment_data = {}
        
        # Duyệt qua các thư mục con (1, 2, 3, 4, 5)
        for sentiment_level in range(1, 6):
            sentiment_path = os.path.join(split_path, str(sentiment_level))
            reviews = []
            
            # Đọc tất cả các file .txt trong thư mục con
            if os.path.exists(sentiment_path):
                for file_name in os.listdir(sentiment_path):
                    file_path = os.path.join(sentiment_path, file_name)
                    if file_path.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            reviews.append(f.read().strip())
            
            sentiment_data[sentiment_level - 1] = reviews
        data[split] = sentiment_data

    return data

## Load configuration file
def load_config(config_path):
    """
    Load configuration file in YAML format.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The configuration file '{config_path}' was not found.")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit(1)
