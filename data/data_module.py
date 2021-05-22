from io import BytesIO
import itertools
import json
import os
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pytorch_lightning as pl
import requests
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast


URL = "https://www.dropbox.com/s/2j9h1yyi8tb6l1w/graph_relations.zip?dl=1"
JSON_PATH = "./json"
DUMP_PATH = "./dumps"
TRAIN_JSON_NAME = "train_graph_relations_bert_3K_rel.json"
TEST_JSON_NAME = "test_graph_relations_bert_3K_rel.json"
TRAIN_DUMP_NAME = "train.pth"
VAL_DUMP_NAME = "val.pth"
TEST_DUMP_NAME = "test.pth"

LEVEL_TO_ID = {
    ("current", "current"): 0,
    ("hypernyms", "current"): 1,
    ("hypernyms", "hypernyms"): 2,
    ("hyponyms", "current"): 3,
    ("hyponyms", "hyponyms"): 4,
    ("hypernyms", "hyponyms"): 5,
    ("hyponyms", "hypernyms"): 6,
}

ENC_TO_ID = {
    "token_ids": 0,
    "level_ids": 1,
    "synset_ids": 2,
    "lemma_ids": 3,
    "is_highway": 4,
}


def where_in(a, b):
    """
    Find where values of a first tensor are equal to values of a second one.

    :param a: torch.tensor, field tensor
    :param b: torch.tensor, query tensor
    :return: torch.tensor, indices where values of b were found in a
    """
    return (a[..., None] == b).any(-1).nonzero().squeeze()


def choose(n, a):
    """
    Randomly choose n elements from a 1d-tensor.

    :param n: int, number of elements to draw
    :param a: torch.tensor, tensor to draw elements from
    """
    return torch.as_tensor([a[idx] for idx in torch.randperm(len(a))[:n]])


class TaxoBERTDataset(Dataset):
    """
    A simple data container for TaxoBERT.

    :param token_ids: torch.tensor, encoded tokens
    :param level_ids: torch.tensor, encoded graph levels
    :param synset_ids: torch.tensor, encoded synsets
    :param is_highway: torch.tensor, contains 1 if the related token is on the
        main branch of the graph, else 0
    :param target_ids: list, lemmas of the masked synset
    """

    def __init__(
        self,
        token_ids,
        level_ids,
        synset_ids,
        lemma_ids,
        is_highway,
        target_ids
    ):
        self.token_ids = token_ids
        self.level_ids = level_ids
        self.synset_ids = synset_ids
        self.lemma_ids = lemma_ids
        self.is_highway = is_highway
        self.target_ids = target_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        inp = (
            self.token_ids[item],
            self.level_ids[item],
            self.synset_ids[item],
            self.lemma_ids[item],
            self.is_highway[item]
        )
        target = self.target_ids[item]

        return inp, target


class TaxoBERTDatasetTrain(TaxoBERTDataset):
    def __init__(self, token_ids, level_ids, synset_ids, lemma_ids, is_highway, target_ids, mask_token_id):
        super().__init__(token_ids, level_ids, synset_ids, lemma_ids, is_highway, target_ids)
        self.mask_token_id = mask_token_id

    def __getitem__(self, item):
        random_lemma = torch.randint(len(self.target_ids[item]), (1, )).item()
        n_masks = len(self.target_ids[item][random_lemma])
        inp = ([self.mask_token_id] * n_masks + self.token_ids[item],
               [0] * n_masks + self.level_ids[item],
               [0] * n_masks + self.synset_ids[item],
               [0] * n_masks + self.lemma_ids[item],
               [True] * n_masks + self.is_highway[item])
        target = self.target_ids[item][random_lemma]
        return inp, target


class TaxoBERTDataModule(pl.LightningDataModule):
    """
    A PyTorch Lighting data module for TaxoBERT.

    :param batch_size: int, the number of samples per batch (default: 32)
    :param val_ratio: float, the ratio of samples from the training set to be
        used for validation (default: 0.1)
    :param max_synsets: int, the number of synsets to sample per level
        (default: 3)
    :param max_lemmas: int, the number of lemmas to sample per synset
        (default: 3)
    :param force_process: bool, if True avoid loading data from dumps
        (default: False)
    :param data_root: str, path to the root with all data files (default: "./")
    """
    def __init__(
        self,
        batch_size: int = 32,
        val_ratio: float = 0.1,
        max_synsets: int = 3,
        max_lemmas: int = 3,
        force_process: bool = False,
        data_root: str = "./",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.force_process = force_process
        self.max_synsets = max_synsets
        self.max_lemmas = max_lemmas

        self.json_path = os.path.join(data_root, JSON_PATH)
        self.train_json_path = os.path.join(self.json_path, TRAIN_JSON_NAME)
        self.test_json_path = os.path.join(self.json_path, TEST_JSON_NAME)
        self.train_dump_path = os.path.join(data_root, DUMP_PATH, TRAIN_DUMP_NAME)
        self.val_dump_path = os.path.join(data_root, DUMP_PATH, VAL_DUMP_NAME)
        self.test_dump_path = os.path.join(data_root, DUMP_PATH, TEST_DUMP_NAME)

        self.level_to_id = LEVEL_TO_ID
        self.enc_to_id = ENC_TO_ID

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            json_obj = json.load(f)

        return json_obj

    def prepare_data(self):
        has_all_files = all(
            os.path.isfile(p) for p in
            (self.train_json_path, self.test_json_path)
        )

        if not has_all_files:
            # Download the archive
            print("Downloading the data")
            r = requests.get(URL, stream=True)

            # Unzip the archive
            print("Extracting the data")
            z = ZipFile(BytesIO(r.content))
            z.extractall(self.json_path)

    def process_data(self, json_dict: dict):
        """
        Unpack the data from a JSON object and create encodings.

        This function assumes that for each synset, its lemma on highway is
        always at index 0.

        :param json_dict: dict, graph structured data read from a JSON file
        :return: tuple, all lemmas and their relevant encodings
        """
        all_token_ids = []
        all_level_ids = []
        all_synset_ids = []
        all_lemma_ids = []
        all_is_highway = []
        all_targets = []

        def tokenize(lemma_):
            return self.tokenizer(
                lemma_,
                add_special_tokens=False,
                truncation=True,
                is_split_into_words=True,
                return_token_type_ids=False,
            ).input_ids

        def add_lemma(lemma_, abs_level_, synset_id_, is_highway_):
            lemma_token_ids = tokenize([lemma_])
            n_tokens_ = len(lemma_token_ids)
            token_ids.extend(lemma_token_ids)
            level_ids.extend([self.level_to_id[abs_level_]] * n_tokens_)
            synset_ids.extend([synset_id_] * n_tokens_)
            lemma_ids.extend([lemma_ids[-1] + 1] * n_tokens_)
            is_highway.extend([is_highway_] * n_tokens_)

        # Go through all JSON entries
        for synset in tqdm(json_dict.values()):
            token_ids = []
            level_ids = []
            synset_ids = [0]
            lemma_ids = [0]
            is_highway = []

            lemmas = [l.replace("_", " ") for l in synset["lemmas"]]
            abs_level = ("current", "current")

            # Save all lemmas of the current node
            synset_token_ids = self.tokenizer.batch_encode_plus(lemmas,
                                                                add_special_tokens=False,
                                                                return_token_type_ids=False).input_ids
            all_targets.append(synset_token_ids)

            for level in ("hypernyms", "hyponyms"):
                for sub_synset in synset[level].values():
                    if "lemmas" in sub_synset:
                        lemmas = [l.replace("_", " ") for l in sub_synset["lemmas"]]
                        abs_level = (level, "current")
                        synset_id = synset_ids[-1] + 1

                        # Add the synset's lemma that is on highway
                        highway_lemma = lemmas.pop(0)
                        add_lemma(highway_lemma, abs_level, synset_id, True)

                        # Add the synset's other lemmas
                        for lemma in lemmas:
                            add_lemma(lemma, abs_level, synset_id, False)

                    for sub_level in ("hypernyms", "hyponyms"):
                        for sub_sub_lemmas in sub_synset[sub_level].values():
                            lemmas = [l.replace("_", " ") for l in sub_sub_lemmas]
                            abs_level = (level, sub_level)
                            synset_id = synset_ids[-1] + 1

                            # Add the synset's lemma that is on highway
                            highway_lemma = lemmas.pop(0)
                            add_lemma(highway_lemma, abs_level, synset_id, True)

                            # Add the synset's other lemmas
                            for lemma in lemmas:
                                add_lemma(lemma, abs_level, synset_id, False)

            # Append the global lists
            all_token_ids.append(token_ids)
            all_level_ids.append(level_ids)
            all_synset_ids.append(synset_ids[1:])
            all_lemma_ids.append(lemma_ids[1:])
            all_is_highway.append(is_highway)

        data = (
            all_token_ids,
            all_level_ids,
            all_synset_ids,
            all_lemma_ids,
            all_is_highway,
            all_targets
        )

        return data

    def setup(self, stage: Optional[str] = None):

        if stage in (None, 'fit'):
            if Path(self.train_dump_path).exists() and not self.force_process:
                print("Loading the training data")
                self.train_set = torch.load(self.train_dump_path)
                self.val_set = torch.load(self.val_dump_path)
            else:
                print("Processing the training data")
                train_val_json = self.read_json(self.train_json_path)
                data = self.process_data(train_val_json)
                # Use a part of the training set as validation set
                full_len = len(data[0])
                val_share = int(self.val_ratio * full_len)
                torch.manual_seed(0)
                indices_order = torch.randperm(full_len)
                self.train_set = TaxoBERTDatasetTrain(*[[item for i, item in enumerate(data_list)
                                                         if i in indices_order[:-val_share]] for data_list in data],
                                                      self.tokenizer.mask_token_id)
                self.val_set = TaxoBERTDataset(*[[item for i, item in enumerate(data_list)
                                                  if i in indices_order[-val_share:]] for data_list in data])
                save_path = Path(self.train_dump_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.train_set, save_path)
                torch.save(self.val_set, Path(self.val_dump_path))

        if stage in (None, 'test'):
            if Path(self.test_dump_path).exists() and not self.force_process:
                print("Loading the test data")
                self.test_set = torch.load(self.test_dump_path)
            else:
                print("Processing the test data")
                test_json = self.read_json(self.test_json_path)
                data = self.process_data(test_json)
                self.test_set = TaxoBERTDataset(*data)
                save_path = Path(self.test_dump_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.test_set, save_path)

    def collate_fn(self, batch):
        # Extract inputs and targets
        x, y = list(zip(*batch))
        enc_lists = [[torch.as_tensor(enc) for enc in seq] for seq in zip(*x)]

        # Randomly sample sequences
        if self.max_synsets and self.max_lemmas:
            sampled_enc_lists = []

            for seq in zip(*enc_lists):
                # Initialize sampled indices with those of the base level
                samples = torch.nonzero(
                    seq[self.enc_to_id["level_ids"]]
                    == self.level_to_id[("current", "current")],
                    as_tuple=True
                )[0]

                for level_id in list(self.level_to_id.values())[1:]:

                    # Sample some synsets
                    level_mask = seq[self.enc_to_id["level_ids"]] == level_id
                    level_synsets = seq[
                        self.enc_to_id["synset_ids"]
                    ][level_mask].unique()
                    kept_synsets = choose(self.max_synsets, level_synsets)

                    # Sample lemmas in those synsets
                    for synset_id in kept_synsets:
                        synset_mask = seq[self.enc_to_id["synset_ids"]] \
                            == synset_id
                        synset_lemmas = seq[
                            self.enc_to_id["lemma_ids"]
                        ][synset_mask].unique()
                        kept_lemmas = choose(self.max_lemmas, synset_lemmas)
                        lemma_indices = where_in(
                            seq[self.enc_to_id["lemma_ids"]], kept_lemmas
                        )
                        samples = torch.cat((samples, lemma_indices.flatten()))

                # Keep the sampled encodings
                new_seq = [
                    seq[index][samples.long()]
                    for index in self.enc_to_id.values()
                ]
                sampled_enc_lists.append(new_seq)

            enc_lists = list(zip(*sampled_enc_lists))

        # Flatten targets of the batch
        if type(y[0][0]) is not list:
            y = torch.as_tensor(list(itertools.chain.from_iterable(y)))

        # Pad encodings (really do pad them all with 0's?)
        padded = [
            pad_sequence(
                encs,
                batch_first=True,
                padding_value=0
            ) for encs in enc_lists
        ]

        return padded, y

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    dm = TaxoBERTDataModule(force_process=True)
    dm.prepare_data()
    dm.setup("test")
    batch = next(iter(dm.test_dataloader()))
    torch.set_printoptions(profile="full")

    print("*" * 80)
    print("Tokens:")
    print(" ".join(dm.tokenizer.convert_ids_to_tokens(batch[0][0][0])))
    id_to_enc = {i: e for e, i in dm.enc_to_id.items()}
    for i_enc, enc in enumerate(batch[0][0]):
        print("*" * 80)
        print(id_to_enc[i_enc])
        print(enc)
    print("*" * 80)
    print("Batch target:")
    print(batch[1])
