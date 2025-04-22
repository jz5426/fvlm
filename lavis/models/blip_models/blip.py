"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
from packaging import version

import torch
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from transformers import BertTokenizer
import transformers

class BlipBase(BaseModel):
    def __init__(self):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version < version.parse("4.27"), "BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade"
        
    @classmethod
    def init_tokenizer(cls):
        # tokenizer = BertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")
        tokenizer = BertTokenizer.from_pretrained(
            '/cluster/projects/mcintoshgroup/fvlm_files/fvlm_weights/BiomedVLP-CXR-BERT-specialized',
            config='/cluster/projects/mcintoshgroup/fvlm_files/fvlm_weights/BiomedVLP-CXR-BERT-specialized/config.json', 
        )
        # TODO: 
        return tokenizer

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], self.visual_encoder
        )
        if "visual_encoder_m.pos_embed" in self.state_dict().keys():
            state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
                state_dict["visual_encoder_m.pos_embed"], self.visual_encoder_m
            )

        for key in self.state_dict().keys():
            if key in state_dict.keys():
                if state_dict[key].shape != self.state_dict()[key].shape:
                    del state_dict[key]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg
    

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print(
            "reshape position embedding from %d to %d" % (orig_size**2, new_size**2)
        )

        return new_pos_embed
    else:
        return pos_embed_checkpoint

