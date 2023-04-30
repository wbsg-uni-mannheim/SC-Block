# collator for pair-wise cross-entropy fine-tuning
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from typing import Optional


@dataclass
class CrossEncoderDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, input):
        features = [x['features'] for x in input]
        #features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]

        batch = self.tokenizer(features, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors=self.return_tensors)
        #batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length,
        #                             return_tensors=self.return_tensors)

        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        #batch['input_ids_right'] = batch_right['input_ids']
        #batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch