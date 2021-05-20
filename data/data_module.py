from io import BytesIO
import json
import os
from pathlib import Path
from string import punctuation
from typing import Optional
from zipfile import ZipFile

import pytorch_lightning as pl
import requests
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast


URL = "https://www.dropbox.com/s/jcx2ld4jw5tbvrb/taxo_bert_test3K.zip?dl=1"
JSON_PATH = "./json"
DUMP_PATH = "./dumps"
TRAIN_JSON_NAME = "train_graph_relations_bert_3K.json"
TEST_JSON_NAME = "test_graph_relations_bert_3K.json"
TRAIN_DUMP_NAME = "train.pth"
TEST_DUMP_NAME = "test.pth"

LEVEL_TO_ID = {
    "lemmas": 0,
    "hypernyms": 1,
    "second_order_hypernyms": 2,
    "hyponyms": 3,
    "second_order_hyponyms": 4,
    "co_hypernyms": 5,
    "co_hyponyms": 6,
}


class TaxoBERTDataset(Dataset):
    """
    A simple data container for TaxoBERT.

    :param token_ids: torch.tensor, encoded tokens
    :param level_ids: torch.tensor, encoded graph levels
    :param synset_ids: torch.tensor, encoded synsets
    :param is_highway: torch.tensor, contains 1 if the related token is on the
        main branch of the graph, else 0
    :param lemmas: list, lemmas of the masked synset
    """
    def __init__(self, token_ids, level_ids, synset_ids, is_highway, target_ids):
        self.token_ids = token_ids
        self.level_ids = level_ids
        self.synset_ids = synset_ids
        self.is_highway = is_highway
        self.target_ids = target_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        inp = (
            self.token_ids[item],
            self.level_ids[item],
            self.synset_ids[item],
            self.is_highway[item]
        )
        target = self.target_ids[item]

        return inp, target


class TaxoBERTDataModule(pl.LightningDataModule):
    """
    A PyTorch Lighting data module for TaxoBERT.

    :param batch_size: int, the number of samples per batch (default: 32)
    :param val_ratio: float, the ratio of samples from the training set to be
        used for validation (default: 0.1)
    :param force_format: bool, if True avoid loading data from dumps
        (default: False)
    """
    def __init__(self,
                 batch_size: int = 32,
                 val_ratio: float = 0.1,
                 force_format: bool = False
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.force_format = force_format

        # TODO: Make those arguments
        self.train_json_path = JSON_PATH + os.sep + TRAIN_JSON_NAME
        self.test_json_path = JSON_PATH + os.sep + TEST_JSON_NAME
        self.train_dump_path = DUMP_PATH + os.sep + TRAIN_DUMP_NAME
        self.test_dump_path = DUMP_PATH + os.sep + TEST_DUMP_NAME
        self.level_to_id = LEVEL_TO_ID

        # Instantiate the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            json_obj = json.load(f)

        return json_obj

    def prepare_data(self):
        has_all_files = all(os.path.isfile(p) \
            for p in (self.train_json_path, self.test_json_path)
        )

        if not has_all_files:
            # Download the archive
            print("Downloading the data")
            r = requests.get(URL, stream=True)

            # Unzip the archive
            print("Extracting the data")
            zip = ZipFile(BytesIO(r.content))
            zip.extractall(JSON_PATH)

    def process_data(self, json_obj):
        """
        Seems to be OK, but time and memory demanding
        """
        all_token_ids = []
        all_level_ids = []
        all_synset_ids = []
        all_is_highway = []
        all_target_ids = []
        levels = [l for l in self.level_to_id.keys() if l != "lemmas"]

        # Go through all JSON entries
        for synset, entry in tqdm(json_obj.items()):
            token_ids = []
            level_ids = []
            synset_ids = []
            is_highway = []

            # Get the ground truth tokens, and mask them
            clean_synset = synset.split(".")[0].replace("_", " ")
            synset_token_ids = self.tokenizer(
                clean_synset,
                add_special_tokens=False,
                truncation=True
            ).input_ids
            n_tokens = len(synset_token_ids)
            all_target_ids.append(synset_token_ids)
            token_ids.extend([self.tokenizer.mask_token_id] * n_tokens)
            level_ids.extend([self.level_to_id["lemmas"]] * n_tokens)
            synset_ids.extend([0] * n_tokens)
            is_highway.extend([True] * n_tokens)

            # Add neighbors
            for level in levels:
                for sub_synset, lemmas in entry[level].items():
                    # Tokenize the synset
                    clean_synset = sub_synset.split(".")[0].replace("_", " ")
                    synset_token_ids = self.tokenizer(
                        clean_synset,
                        add_special_tokens=False,
                        truncation=True
                    ).input_ids
                    n_tokens = len(synset_token_ids)
                    token_ids.extend(synset_token_ids)
                    level_ids.extend([self.level_to_id[level]] * n_tokens)
                    synset_id = synset_ids[-1] + 1
                    synset_ids.extend([synset_id] * n_tokens)
                    is_highway.extend([True] * n_tokens)

                    # Tokenize the synset's other lemmas
                    clean_lemmas = " ".join(lemmas).replace("_", " ")
                    clean_lemmas = clean_lemmas.replace(clean_synset, "")
                    lemmas_token_ids = self.tokenizer(
                        clean_lemmas,
                        truncation=True,
                        add_special_tokens=False
                    ).input_ids
                    n_tokens = len(lemmas_token_ids)
                    token_ids.extend(lemmas_token_ids)
                    level_ids.extend([self.level_to_id[level]] * n_tokens)
                    synset_ids.extend([synset_id] * n_tokens)
                    is_highway.extend([False] * n_tokens)

            # Append the global lists
            all_token_ids.append(torch.as_tensor(token_ids))
            all_level_ids.append(torch.as_tensor(level_ids))
            all_synset_ids.append(torch.as_tensor(synset_ids))
            all_is_highway.append(torch.as_tensor(is_highway))

        # Pad all sequences
        all_token_ids = pad_sequence(
            all_token_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        all_level_ids = pad_sequence(
            all_level_ids,
            batch_first=True,
            padding_value=-1
        )
        all_synset_ids = pad_sequence(
            all_synset_ids,
            batch_first=True,
            padding_value=-1
        )
        all_is_highway = pad_sequence(
            all_is_highway,
            batch_first=True,
            padding_value=False
        )

        data = (
            all_token_ids,
            all_level_ids,
            all_synset_ids,
            all_is_highway,
            all_target_ids
        )

        return data

    def recover_encodings(self, encodings, lists):
        all_starts = encodings.offset_mapping[:, :, 0]
        max_len = all_starts.size()[-1]
        recovered = [torch.zeros_like(encodings.input_ids) for l in lists]

        for i, starts in enumerate(tqdm(all_starts[:, 1:])) :
            j = 0
            k = 0
            repeats = [1]

            while len(repeats) < len(lists[0][i]) and k < max_len - 1:
                if starts[k] != 0:
                    # val += 1
                    repeats[-1] += 1
                else:
                    repeats.append(1)
                    j += 1
                k += 1

            repeats = torch.as_tensor(repeats)
            length = repeats.sum().item()

            for i_l, l in enumerate(lists):
                t = torch.repeat_interleave(
                    torch.as_tensor(l[i][:len(repeats)]), repeats
                )
                recovered[i_l][i, :length] = t

        return recovered

    def process_data_fast(self, json_obj):
        """
        TODO: Check outputs
        """
        all_to_tokenize = []
        all_level_ids = []
        all_synset_ids = []
        all_is_highway = []
        all_targets = []
        levels = [l for l in self.level_to_id.keys() if l != "lemmas"]

        # Go through all JSON entries
        for synset, entry in tqdm(json_obj.items()):
            to_tokenize = []
            level_ids = []
            synset_ids = []
            is_highway = []

            # Get the ground truth tokens, and mask them
            clean_synset = "".join(synset.split(".")[:-2]).split("_")
            len_synset = len(clean_synset)
            all_targets.append(clean_synset)
            to_tokenize.extend([self.tokenizer.mask_token] * len_synset)
            level_ids.extend([self.level_to_id["lemmas"]] * len_synset)
            synset_ids.extend([0] * len_synset)
            is_highway.extend([True] * len_synset)

            # Add neighbors
            for level in levels:
                for sub_synset, lemmas in entry[level].items():
                    # Tokenize the synset
                    clean_synset = "".join(sub_synset.split(".")[:-2]).lower()
                    clean_synset = clean_synset.replace(".", "").split("_")
                    len_synset = len(clean_synset)
                    to_tokenize.extend(clean_synset)
                    level_ids.extend([self.level_to_id[level]] * len_synset)
                    synset_id = synset_ids[-1] + 1
                    synset_ids.extend([synset_id] * len_synset)
                    is_highway.extend([True] * len_synset)

                    # Tokenize the synset's other lemmas
                    clean_lemmas = [
                        l.lower().replace(".", "").split("_") for l in lemmas
                    ]
                    clean_lemmas = [l for l_s in clean_lemmas for l in l_s]
                    for s in clean_synset:
                        clean_lemmas.remove(s)
                    n_lemmas = len(clean_lemmas)
                    to_tokenize.extend(clean_lemmas)
                    level_ids.extend([self.level_to_id[level]] * n_lemmas)
                    synset_ids.extend([synset_id] * n_lemmas)
                    is_highway.extend([False] * n_lemmas)

            # Append the global lists
            all_to_tokenize.append(to_tokenize)
            all_level_ids.append(level_ids)
            all_synset_ids.append(synset_ids)
            all_is_highway.append(is_highway)

        # Perform tokenization
        print("Performing tokenization")
        inp_encodings = self.tokenizer(
            all_to_tokenize,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
            return_token_type_ids=False,
            return_offsets_mapping=True
        )
        all_token_ids = inp_encodings.input_ids

        print(all_targets)

        out_encodings = self.tokenizer(
            all_targets,
            add_special_tokens=False,
            truncation=True,
            is_split_into_words=True,
            return_token_type_ids=False,
        )
        all_target_ids = out_encodings.input_ids

        # Respan encoding values depending on truncation
        print("Recovering encodings")
        all_level_ids, all_synset_ids, all_is_highway = self.recover_encodings(
            inp_encodings,
            (all_level_ids, all_synset_ids, all_is_highway)
        )

        data = (
            all_token_ids,
            all_level_ids,
            all_synset_ids,
            all_is_highway,
            all_target_ids
        )

        return data

    def setup(self, stage: Optional[str] = None):

        if stage in (None, 'fit'):
            if Path(self.train_dump_path).exists() and not self.force_format:
                print("Loading the training data")
                train_val_set = torch.load(self.train_dump_path)
            else:
                print("Processing the training data")
                train_val_json = self.read_json(self.train_json_path)
                # TODO: Check self.process_data_fast
                data = self.process_data(train_val_json)
                train_val_set = TaxoBERTDataset(*data)
                save_path = Path(self.train_dump_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(train_val_set, save_path)

            # Use a part of the training set as validation set
            full_len = len(train_val_set)
            val_share = self.val_ratio * full_len
            split = [full_len - val_share, val_share]
            self.train_set, self.val_set = random_split(
                train_val_set,
                split,
            )

        if stage in (None, 'test'):
            if Path(self.test_dump_path).exists() and not self.force_format:
                print("Loading the test data")
                self.test_set = torch.load(self.test_dump_path)
            else:
                print("Processing the test data")
                test_json = self.read_json(self.test_json_path)
                # TODO: Check self.process_data_fast
                data = self.process_data(test_json)
                self.test_set = TaxoBERTDataset(*data)
                save_path = Path(self.test_dump_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.test_set, save_path)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


if __name__ == "__main__":
    # For debugging purpose
    dm = TaxoBERTDataModule(force_format=True)
    dm.prepare_data()
    dm.setup("test")

    torch.set_printoptions(profile="full")
    max_len = 50

    print(dm.tokenizer.convert_ids_to_tokens(dm.test_set[0][0][0])[:max_len])
    print("*" * 20)
    for item in dm.test_set[0][0]:
        print(item[:max_len])
        print("*" * 20)

    print("Target:", dm.test_set[0][1])
