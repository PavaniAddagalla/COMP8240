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


class WikipediaTopicMix_Dataset(TorchnlpDataset):

    def __init__(self, root: str, normal_class=0, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        # ---- Available categories (folder names) ----
        data_dir = os.path.join(root, 'data_wiki_mix')
        classes = sorted([d for d in os.listdir(os.path.join(data_dir, 'train')) 
                          if os.path.isdir(os.path.join(data_dir, 'train', d))])

        # ---- Define normal and outlier classes ----
        if normal_class == -1:
            self.normal_classes = classes
            self.outlier_classes = []
        else:
            if isinstance(normal_class, int) and 0 <= normal_class < len(classes):
                self.normal_classes = [classes[normal_class]]
            elif isinstance(normal_class, str) and normal_class in classes:
                self.normal_classes = [normal_class]
            else:
                self.normal_classes = [classes[0]]
            self.outlier_classes = [c for c in classes if c not in self.normal_classes]

        # ---- Load dataset from folders ----
        self.train_set, self.test_set = wikipedia_topic_mix_dataset(directory=root, train=True, test=True, clean_txt=clean_txt)

        # ---- Preprocess: add columns ----
        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('label')
        self.test_set.columns.add('label')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')

        # ---- Assign labels ----
        train_idx_normal = []
        for i, row in enumerate(self.train_set):
            row['label'] = row.pop('category')
            if row['label'] in self.normal_classes:
                train_idx_normal.append(i)
                row['label'] = torch.tensor(0)
            else:
                row['label'] = torch.tensor(1)
            row['text'] = row['text'].lower()
            if clean_txt:
                row['text'] = clean_text(row['text'])

        for i, row in enumerate(self.test_set):
            row['label'] = row.pop('category')
            row['label'] = torch.tensor(0) if row['label'] in self.normal_classes else torch.tensor(1)
            row['text'] = row['text'].lower()
            if clean_txt:
                row['text'] = clean_text(row['text'])

        # ---- Subset train_set to normal samples only ----
        self.train_set = Subset(self.train_set, train_idx_normal)

        # ---- Build corpus and encoder ----
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        if tokenizer == 'spacy':
            self.encoder = SpacyEncoder(text_corpus, min_occurrences=3, append_eos=append_eos)
        elif tokenizer == 'bert':
            self.encoder = MyBertTokenizer.from_pretrained('bert-base-uncased', cache_dir=root)

        # ---- Encode text ----
        for row in datasets_iterator(self.train_set, self.test_set):
            if append_sos:
                sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
                row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), self.encoder.encode(row['text'])))
            else:
                row['text'] = self.encoder.encode(row['text'])

        # ---- Compute TF-IDF weights ----
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.test_set):
                row['weight'] = torch.empty(0)

        # ---- Add index column ----
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i


def wikipedia_topic_mix_dataset(directory='../data', train=False, test=False, clean_txt=False):
    """
    Reads the Wikipedia Topic Mix dataset from the following structure:
        data_wiki_mix/
            train/
                Astronomy/
                Cooking/
                ...
            test/
                Astronomy/
                Cooking/
                ...
    Returns: torchnlp Dataset objects (train/test)
    """

    def load_from_folders(base_dir):
        examples = []
        if not os.path.exists(base_dir):
            return examples
        for category in sorted(os.listdir(base_dir)):
            cat_path = os.path.join(base_dir, category)
            if not os.path.isdir(cat_path):
                continue
            for fname in os.listdir(cat_path):
                if fname.endswith('.txt'):
                    with open(os.path.join(cat_path, fname), "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if clean_txt:
                        text = clean_text(text)
                    examples.append({'text': text, 'category': category})
        return examples

    data_dir = os.path.join(directory, 'data_wiki_mix')
    ret = []
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

    for split_set in splits:
        base_dir = os.path.join(data_dir, split_set)
        examples = load_from_folders(base_dir)
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
