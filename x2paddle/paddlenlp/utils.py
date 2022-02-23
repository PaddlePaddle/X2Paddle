# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import io
import json
import os
import six
import inspect
from collections import OrderedDict

import torch
import paddle


def convert_weight_from_hf(weight_path, class_name):
    """
    Args:
    weight_path (str): HF weight file path
    class_name (str): The class name used by the user
    Return:
    paddle_state_dict (dict): PaddleNLP state_dict
    """
    pytorch_state_dict = torch.load(weight_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    hf_to_paddle = {
        "embeddings.LayerNorm": "embeddings.layer_norm",
        "encoder.layer": "encoder.layers",
        "attention.self.query": "self_attn.q_proj",
        "attention.self.key": "self_attn.k_proj",
        "attention.self.value": "self_attn.v_proj",
        "attention.output.dense": "self_attn.out_proj",
        "intermediate.dense": "linear1",
        "output.dense": "linear2",
        "attention.output.LayerNorm": "norm1",
        "output.LayerNorm": "norm2",
        "predictions.decoder.": "predictions.decoder_",
        "predictions.transform.dense": "predictions.transform",
        "predictions.transform.LayerNorm": "predictions.layer_norm",
    }
    for k, v in pytorch_state_dict.items():
        if k[-7:] == ".weight":
            if ".embeddings." not in k and ".LayerNorm." not in k:
                if v.ndim == 2:
                    v = v.transpose(0, 1)
        for hf_name, paddle_name in hf_to_paddle.items():
            k = k.replace(hf_name, paddle_name)

        if "bert." not in k and "cls." not in k and "classifier" not in k:
            k = "bert." + k
        paddle_state_dict[k] = paddle.to_tensor(v.data.numpy())

    return paddle_state_dict


def convert_config_from_hf(config_path, derived_parameters_dict, class_name):
    """
    Args:
    config_path (str): HF config file path
    derived_parameters_dict (dict): The parameter dict required by the init function to initialize
    class_name (str): The class name used by the user
    Return:
    derived_config (dict): PaddleNLP config
    """
    default_config = {
        "vocab_size": 28996,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "init_class": "BertModel"
    }
    with io.open(config_path, encoding="utf-8") as f:
        init_kwargs = json.load(f)
    base_config = default_config
    for k, v in init_kwargs.items():
        if k in base_config:
            base_config[k] = v
    if class_name == "BertModel":
        return base_config
    else:
        derived_config = {"init_args": [base_config], "init_class": class_name}
        for k, v in derived_parameters_dict.items():
            if k == "self" or k == "bert":
                continue
            derived_config[k] = v.default

        for k, v in init_kwargs.items():
            if k in derived_config:
                derived_config[k] = v
        if "id2label" in init_kwargs:
            if "num_classes" in derived_config:
                derived_config["num_classes"] = len(init_kwargs["id2label"])
            elif "num_choices" in derived_config:
                derived_config["num_choices"] = len(init_kwargs["id2label"])
    return derived_config
