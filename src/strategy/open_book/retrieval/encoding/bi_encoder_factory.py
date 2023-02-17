import logging

from src.strategy.open_book.retrieval.encoding.bi_encoder import BiEncoder
from src.strategy.open_book.retrieval.encoding.dl_block_bi_encoder import DLBlockBiEncoder
from src.strategy.open_book.retrieval.encoding.glove_bi_encoder import GloveBiEncoder
from src.strategy.open_book.retrieval.encoding.hf_bi_encoder import HuggingfaceBiEncoder
from src.strategy.open_book.retrieval.encoding.sbert_bi_encoder import SBERTBiEncoder
from src.strategy.open_book.retrieval.encoding.supcon_bi_encoder import SupConBiEncoder
from src.strategy.open_book.retrieval.encoding.word2vec_bi_encoder import Word2VecBiEncoder


def select_bi_encoder(bi_encoder_config, schema_org_class):
    logger = logging.getLogger()
    logger.info('Select Bi Encoder {}!'.format(bi_encoder_config['name']))

    if bi_encoder_config['name'] == 'huggingface_bi_encoder':
        bi_encoder = HuggingfaceBiEncoder(bi_encoder_config['model_name'], bi_encoder_config['pooling'],
                                          bi_encoder_config['normalize'], schema_org_class)
    elif bi_encoder_config['name'] == 'sbert_bi_encoder':
        bi_encoder = SBERTBiEncoder(bi_encoder_config['model_name'], bi_encoder_config['pooling'],
                                          bi_encoder_config['normalize'], schema_org_class)
    elif bi_encoder_config['name'] == 'supcon_bi_encoder':
        bi_encoder = SupConBiEncoder(bi_encoder_config['model_name'], bi_encoder_config['base_model'],
                                     bi_encoder_config['with_projection'], bi_encoder_config['projection'],
                                     bi_encoder_config['pooling'], bi_encoder_config['normalize'], schema_org_class)
    elif bi_encoder_config['name'] == 'word2vec_bi_encoder':
        bi_encoder = Word2VecBiEncoder(bi_encoder_config['model_name'])
    elif bi_encoder_config['name'] == 'glove_bi_encoder':
        bi_encoder = GloveBiEncoder(bi_encoder_config['model_name'])
    elif bi_encoder_config['name'] == 'dl_block_bi_encoder':
        bi_encoder = DLBlockBiEncoder(bi_encoder_config['model_name'], bi_encoder_config['base_model'], schema_org_class)
    else:
        # Fall back to default open book strategy
        bi_encoder = BiEncoder(schema_org_class)

    return bi_encoder
