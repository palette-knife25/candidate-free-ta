from transformers import BertTokenizerFast

from .kbert_enricher import KBertEnricher
from .kbert_gat_enricher import KBertGATEnricher
from .fixed_enricher import FixedTokenizer, FixedEnricher


# noinspection PyPep8Naming
def BertFastTokenizer(model: str = 'bert-base-uncased'):
    return BertTokenizerFast.from_pretrained(model)
