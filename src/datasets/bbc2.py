from base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from utils.text_encoders import MyBertTokenizer
from utils.misc import clean_text
from .preprocessing import compute_tfidf_weights

import torch
import os
import glob
import random


class BBC2_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, clean_txt=False, split_ratio=0.8, seed=1):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        # Define categories (folders inside data/bbc-2)
        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

        # Map normal_class index to categories similar to newsgroups grouping style
        # Support normal_class == -1 meaning all normal
        if normal_class == -1:
            self.normal_classes = categories
            self.outlier_classes = []
        else:
            # allow integer selection or list of names
            if isinstance(normal_class, int):
                assert 0 <= normal_class < len(categories), 'normal_class index out of range'
                self.normal_classes = [categories[normal_class]]
            elif isinstance(normal_class, str):
                self.normal_classes = [normal_class]
            elif isinstance(normal_class, (list, tuple)):
                self.normal_classes = list(normal_class)
            else:
                self.normal_classes = [categories[0]]

            # outlier classes are the remaining categories
            self.outlier_classes = [c for c in categories if c not in self.normal_classes]

        # Build train/test splits by reading files from data/bbc-2/<category>/*.txt
        data_dir = os.path.join(root, 'bbc-2') if os.path.isdir(os.path.join(root, 'bbc-2')) else os.path.join(root, 'data', 'bbc-2')
        all_examples = []
        for cat in categories:
            cat_dir = os.path.join(data_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            files = glob.glob(os.path.join(cat_dir, '*.txt'))
            for f in files:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read().strip()
                    if not text:
                        continue
                    if clean_txt:
                        text = clean_text(text)
                    all_examples.append({'text': text, 'label': cat})

        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_examples)
        split_at = int(len(all_examples) * split_ratio)
        train_examples = all_examples[:split_at]
        test_examples = all_examples[split_at:]

        # Convert to torchnlp Dataset instances
        self.train_set = Dataset(train_examples)
        self.test_set = Dataset(test_examples)

        # Pre-process: add expected columns
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        train_idx_normal = []
        for i, row in enumerate(self.train_set):
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            # keep text lowercase for consistency like other loaders
            row['text'] = row['text'].lower()

        for i, row in enumerate(self.test_set):
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            row['text'] = row['text'].lower()

        # Subset train_set to normal class
        self.train_set = Subset(self.train_set, train_idx_normal)

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        if tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # Encode
        for row in datasets_iterator(self.train_set, self.test_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i


def bbc2_dataset(directory='../data', train=False, test=False, clean_txt=False, split_ratio=0.8, seed=1):
    """
    
    Load the BBC-2 dataset from folders under `data/bbc-2/<category>/*.txt`.

    Returns torchnlp Dataset objects similar to other dataset loaders in this project.
    """

    data_dir = os.path.join(directory, 'bbc-2') if os.path.isdir(os.path.join(directory, 'bbc-2')) else os.path.join(directory, 'data', 'bbc-2')
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

    def read_folder(path):
        examples = []
        if not os.path.isdir(path):
            return examples
        for f in glob.glob(os.path.join(path, '*.txt')):
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read().strip()
                if text:
                    if clean_txt:
                        text = clean_text(text)
                    examples.append({'text': text, 'label': os.path.basename(path)})
        return examples

    all_examples = []
    for cat in categories:
        all_examples += read_folder(os.path.join(data_dir, cat))

    # shuffle and split
    random.seed(seed)
    random.shuffle(all_examples)
    split_at = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_at]
    test_examples = all_examples[split_at:]

    ret = []
    if train:
        ret.append(Dataset(train_examples))
    if test:
        ret.append(Dataset(test_examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

