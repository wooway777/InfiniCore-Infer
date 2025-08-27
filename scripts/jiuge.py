from typing import List, Sequence

from sympy import true
from libinfinicore_infer import (
    JiugeMetaCStruct,
    JiugeWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    create_jiuge_model,
    destroy_jiuge_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
    forward_batch,
)
from infer_task import InferTask, KVCache
from qwen2_tokenizer import piece_to_text

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers
import numpy as np

torch.set_default_device("cpu")


class LlamaWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    def match(self, state_dict):
        # Check for either full precision or quantized weights
        return "model.norm.weight" in state_dict and (
            "model.layers.0.self_attn.q_proj.weight" in state_dict
        )


class GPTQWeightsNaming(LlamaWeightsNaming):
    def attn_q_qweight(self, i):
        return f"model.layers.{i}.self_attn.q_proj.qweight"

    def attn_k_qweight(self, i):
        return f"model.layers.{i}.self_attn.k_proj.qweight"

    def attn_v_qweight(self, i):
        return f"model.layers.{i}.self_attn.v_proj.qweight"

    def attn_o_qweight(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qweight"

    def gate_qweight(self, i):
        return f"model.layers.{i}.mlp.gate_proj.qweight"

    def up_qweight(self, i):
        return f"model.layers.{i}.mlp.up_proj.qweight"

    def down_qweight(self, i):
        return f"model.layers.{i}.mlp.down_proj.qweight"

    def attn_q_scales(self, i):
        return f"model.layers.{i}.self_attn.q_proj.scales"

    def attn_k_scales(self, i):
        return f"model.layers.{i}.self_attn.k_proj.scales"

    def attn_v_scales(self, i):
        return f"model.layers.{i}.self_attn.v_proj.scales"

    def attn_o_scales(self, i):
        return f"model.layers.{i}.self_attn.o_proj.scales"

    def gate_scales(self, i):
        return f"model.layers.{i}.mlp.gate_proj.scales"

    def up_scales(self, i):
        return f"model.layers.{i}.mlp.up_proj.scales"

    def down_scales(self, i):
        return f"model.layers.{i}.mlp.down_proj.scales"

    def attn_q_qzeros(self, i):
        return f"model.layers.{i}.self_attn.q_proj.qzeros"

    def attn_k_qzeros(self, i):
        return f"model.layers.{i}.self_attn.k_proj.qzeros"

    def attn_v_qzeros(self, i):
        return f"model.layers.{i}.self_attn.v_proj.qzeros"

    def attn_o_qzeros(self, i):
        return f"model.layers.{i}.self_attn.o_proj.qzeros"

    def gate_qzeros(self, i):
        return f"model.layers.{i}.mlp.gate_proj.qzeros"

    def up_qzeros(self, i):
        return f"model.layers.{i}.mlp.up_proj.qzeros"

    def down_qzeros(self, i):
        return f"model.layers.{i}.mlp.down_proj.qzeros"

    def attn_q_g_idx(self, i):
        return f"model.layers.{i}.self_attn.q_proj.g_idx"

    def attn_k_g_idx(self, i):
        return f"model.layers.{i}.self_attn.k_proj.g_idx"

    def attn_v_g_idx(self, i):
        return f"model.layers.{i}.self_attn.v_proj.g_idx"

    def attn_o_g_idx(self, i):
        return f"model.layers.{i}.self_attn.o_proj.g_idx"

    def gate_g_idx(self, i):
        return f"model.layers.{i}.mlp.gate_proj.g_idx"

    def up_g_idx(self, i):
        return f"model.layers.{i}.mlp.up_proj.g_idx"

    def down_g_idx(self, i):
        return f"model.layers.{i}.mlp.down_proj.g_idx"

    def match(self, state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.qweight" in state_dict
        )


class JiugeMetaFromLlama(JiugeMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        if (
            config["model_type"] in ["fm9g", "minicpm"]
            and "scale_emb" in config
            and "scale_depth" in config
            and "dim_model_base" in config
        ):
            self.scale_input = config["scale_emb"]
            self.scale_output = config["hidden_size"] // config["dim_model_base"]
            self.scale_o = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )
            self.scale_down = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            end_token=2,
        )
        self.torch_dtype_logits = dtype


class JiugeWeightsImpl(JiugeWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
        is_quantized=False,
        quantization_config=None,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_logits = meta.torch_dtype_logits

        self.is_quantized = is_quantized
        if is_quantized:
            self.bits = quantization_config.get("bits", 8)
            self.group_size = quantization_config.get("group_size", 128)
            self.symmetric = quantization_config.get("sym", True)
            self.dt_qweight = DataType.INFINI_DTYPE_U8
            self.dt_scales = DataType.INFINI_DTYPE_F16
            self.dt_qzeros = DataType.INFINI_DTYPE_U8
            self.dt_g_idx = DataType.INFINI_DTYPE_I32
        else:
            self.bits = 0
            self.group_size = 0
            self.symmetric = False
            self.dt_qweight = DataType.INFINI_DTYPE_INVALID
            self.dt_scales = DataType.INFINI_DTYPE_INVALID
            self.dt_qzeros = DataType.INFINI_DTYPE_INVALID
            self.dt_g_idx = DataType.INFINI_DTYPE_INVALID

        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # Initialize quantized weights if needed
        if is_quantized and isinstance(naming, GPTQWeightsNaming):
            if self.bits == 8:
                self._init_quantized_weights_i8(
                    naming,
                    state_dict,
                    nlayer,
                    ndev,
                    nh,
                    nkvh,
                    dh,
                    d,
                    di,
                    transpose_weight,
                    scale_o,
                    scale_down,
                )
        else:
            self._init_full_precision_weights(
                naming,
                state_dict,
                nlayer,
                ndev,
                nh,
                nkvh,
                dh,
                d,
                di,
                torch_dt_mat,
                torch_dt_logits,
                transpose_weight,
                scale_o,
                scale_down,
            )

        # Common initialization for both quantized and full-precision
        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

    def _init_quantized_weights_i8(
        self,
        naming,
        state_dict,
        nlayer,
        ndev,
        nh,
        nkvh,
        dh,
        d,
        di,
        transpose_weight,
        scale_o,
        scale_down,
    ):
        """Initialize quantized weights with RoPE-friendly QKV concatenation"""
        # Initialize all pointers
        self.attn_qkv_qweight = (c_void_p * nlayer)()
        self.attn_qkv_scales = (c_void_p * nlayer)()
        self.attn_qkv_qzeros = (c_void_p * nlayer)()
        self.attn_qkv_g_idx = (c_void_p * nlayer)()
        self.attn_o_qweight = (c_void_p * nlayer)()
        self.attn_o_scales = (c_void_p * nlayer)()
        self.attn_o_qzeros = (c_void_p * nlayer)()
        self.attn_o_g_idx = (c_void_p * nlayer)()
        self.ffn_gate_up_qweight = (c_void_p * nlayer)()
        self.ffn_gate_up_scales = (c_void_p * nlayer)()
        self.ffn_gate_up_qzeros = (c_void_p * nlayer)()
        self.ffn_gate_up_g_idx = (c_void_p * nlayer)()
        self.ffn_down_qweight = (c_void_p * nlayer)()
        self.ffn_down_scales = (c_void_p * nlayer)()
        self.ffn_down_qzeros = (c_void_p * nlayer)()
        self.ffn_down_g_idx = (c_void_p * nlayer)()

        # Temporary lists to hold tensors
        attn_qkv_qweights = []
        attn_qkv_scales_list = []
        attn_qkv_qzeros_list = []
        attn_qkv_g_idx_list = []
        attn_o_qweights = []
        attn_o_scales_list = []
        attn_o_qzeros_list = []
        attn_o_g_idx_list = []
        ffn_gate_up_qweights = []
        ffn_gate_up_scales_list = []
        ffn_gate_up_qzeros_list = []
        ffn_gate_up_g_idx_list = []
        ffn_down_qweights = []
        ffn_down_scales_list = []
        ffn_down_qzeros_list = []
        ffn_down_g_idx_list = []

        for i in range(nlayer):
            # --- Process QKV weights ---
            # Q weights
            q_qweight = state_dict[naming.attn_q_qweight(i)]
            q_qweight = q_qweight.reshape([nh, 2, dh // 2, -1]).transpose(
                1, 2
            )  # (nh, dh//2, 2, d)
            q_qweight = q_qweight.reshape(-1, q_qweight.shape[-1])  # (nh*dh//2 * 2, d)

            # K weights
            k_qweight = state_dict[naming.attn_k_qweight(i)]
            k_qweight = k_qweight.reshape([nkvh, 2, dh // 2, -1]).transpose(
                1, 2
            )  # (nkvh, dh//2, 2, d)
            k_qweight = k_qweight.reshape(-1, k_qweight.shape[-1])

            # V weights
            v_qweight = state_dict[naming.attn_v_qweight(i)]
            v_qweight = v_qweight.reshape([nkvh, dh // 2, 2, -1])  # (nkvh, dh//2, 2, d)
            v_qweight = v_qweight.reshape(-1, v_qweight.shape[-1])

            # Concatenate QKV
            qkv_qweight = torch.cat(
                [q_qweight, k_qweight, v_qweight], dim=0
            ).contiguous()
            attn_qkv_qweights.append(qkv_qweight)

            # Q scales/qzeros/g_idx
            q_scales = state_dict[naming.attn_q_scales(i)]
            q_scales = (
                q_scales.reshape([nh, 2, dh // 2, -1])
                .transpose(1, 2)
                .reshape(-1, q_scales.shape[-1])
            )
            q_qzeros = state_dict[naming.attn_q_qzeros(i)]
            q_qzeros = (
                q_qzeros.reshape([nh, 2, dh // 2, -1])
                .transpose(1, 2)
                .reshape(-1, q_qzeros.shape[-1])
            )
            q_g_idx = state_dict[naming.attn_q_g_idx(i)]
            q_g_idx = q_g_idx.reshape([nh, 2, dh // 2]).transpose(1, 2).flatten()

            # K scales/qzeros/g_idx
            k_scales = state_dict[naming.attn_k_scales(i)]
            k_scales = (
                k_scales.reshape([nkvh, 2, dh // 2, -1])
                .transpose(1, 2)
                .reshape(-1, k_scales.shape[-1])
            )
            k_qzeros = state_dict[naming.attn_k_qzeros(i)]
            k_qzeros = (
                k_qzeros.reshape([nkvh, 2, dh // 2, -1])
                .transpose(1, 2)
                .reshape(-1, k_qzeros.shape[-1])
            )
            k_g_idx = state_dict[naming.attn_k_g_idx(i)]
            k_g_idx = k_g_idx.reshape([nkvh, 2, dh // 2]).transpose(1, 2).flatten()

            # V scales/qzeros/g_idx
            v_scales = state_dict[naming.attn_v_scales(i)]
            v_scales = v_scales.reshape([nkvh, dh // 2, 2, -1]).reshape(
                -1, v_scales.shape[-1]
            )
            v_qzeros = state_dict[naming.attn_v_qzeros(i)]
            v_qzeros = v_qzeros.reshape([nkvh, dh // 2, 2, -1]).reshape(
                -1, v_qzeros.shape[-1]
            )
            v_g_idx = state_dict[naming.attn_v_g_idx(i)]
            v_g_idx = v_g_idx.reshape([nkvh, dh // 2, 2]).flatten()

            # Concatenate QKV scales/qzeros/g_idx
            qkv_scales = torch.cat([q_scales, k_scales, v_scales], dim=0).contiguous()
            qkv_qzeros = torch.cat([q_qzeros, k_qzeros, v_qzeros], dim=0).contiguous()
            qkv_g_idx = torch.cat([q_g_idx, k_g_idx, v_g_idx], dim=0).contiguous()

            attn_qkv_scales_list.append(qkv_scales)
            attn_qkv_qzeros_list.append(qkv_qzeros)
            attn_qkv_g_idx_list.append(qkv_g_idx)

            # --- Process O weights ---
            o_qweight = state_dict[naming.attn_o_qweight(i)]
            if transpose_weight:
                o_qweight = o_qweight.reshape([d, ndev, nh // ndev * dh]).transpose(
                    0, 1
                )
            o_qweight = o_qweight.contiguous()
            attn_o_qweights.append(o_qweight)

            o_scales = state_dict[naming.attn_o_scales(i)] * scale_o
            if transpose_weight:
                o_scales = o_scales.reshape([d, ndev, nh // ndev * dh]).transpose(0, 1)
            o_scales = o_scales.contiguous()
            attn_o_scales_list.append(o_scales)

            o_qzeros = state_dict[naming.attn_o_qzeros(i)]
            if transpose_weight:
                o_qzeros = o_qzeros.reshape([d, ndev, nh // ndev * dh]).transpose(0, 1)
            o_qzeros = o_qzeros.contiguous()
            attn_o_qzeros_list.append(o_qzeros)

            o_g_idx = state_dict[naming.attn_o_g_idx(i)]
            if transpose_weight:
                o_g_idx = o_g_idx.reshape([d, ndev, nh // ndev * dh]).transpose(0, 1)
            o_g_idx = o_g_idx.contiguous()
            attn_o_g_idx_list.append(o_g_idx)

            # --- Process FFN Gate/Up weights ---
            gate_qweight = state_dict[naming.gate_qweight(i)]
            up_qweight = state_dict[naming.up_qweight(i)]

            if not transpose_weight:
                gate_qweight = gate_qweight.reshape([ndev, di // ndev, d]).transpose(
                    1, 2
                )
                up_qweight = up_qweight.reshape([ndev, di // ndev, d]).transpose(1, 2)

            gate_up_qweight = torch.cat([gate_qweight, up_qweight], dim=0).contiguous()
            ffn_gate_up_qweights.append(gate_up_qweight)

            gate_scales = state_dict[naming.gate_scales(i)]
            up_scales = state_dict[naming.up_scales(i)]
            if not transpose_weight:
                gate_scales = gate_scales.reshape([ndev, di // ndev, d]).transpose(1, 2)
                up_scales = up_scales.reshape([ndev, di // ndev, d]).transpose(1, 2)
            gate_up_scales = torch.cat([gate_scales, up_scales], dim=0).contiguous()
            ffn_gate_up_scales_list.append(gate_up_scales)

            gate_qzeros = state_dict[naming.gate_qzeros(i)]
            up_qzeros = state_dict[naming.up_qzeros(i)]
            if not transpose_weight:
                gate_qzeros = gate_qzeros.reshape([ndev, di // ndev, d]).transpose(1, 2)
                up_qzeros = up_qzeros.reshape([ndev, di // ndev, d]).transpose(1, 2)
            gate_up_qzeros = torch.cat([gate_qzeros, up_qzeros], dim=0).contiguous()
            ffn_gate_up_qzeros_list.append(gate_up_qzeros)

            gate_g_idx = state_dict[naming.gate_g_idx(i)]
            up_g_idx = state_dict[naming.up_g_idx(i)]
            if not transpose_weight:
                gate_g_idx = gate_g_idx.reshape([ndev, di // ndev, d]).transpose(1, 2)
                up_g_idx = up_g_idx.reshape([ndev, di // ndev, d]).transpose(1, 2)
            gate_up_g_idx = torch.cat([gate_g_idx, up_g_idx], dim=0).contiguous()
            ffn_gate_up_g_idx_list.append(gate_up_g_idx)

            # --- Process FFN Down weights ---
            down_qweight = state_dict[naming.down_qweight(i)]
            if transpose_weight:
                down_qweight = down_qweight.reshape([d, ndev, di // ndev]).transpose(
                    0, 1
                )
            down_qweight = down_qweight.contiguous()
            ffn_down_qweights.append(down_qweight)

            down_scales = state_dict[naming.down_scales(i)] * scale_down
            if transpose_weight:
                down_scales = down_scales.reshape([d, ndev, di // ndev]).transpose(0, 1)
            down_scales = down_scales.contiguous()
            ffn_down_scales_list.append(down_scales)

            down_qzeros = state_dict[naming.down_qzeros(i)]
            if transpose_weight:
                down_qzeros = down_qzeros.reshape([d, ndev, di // ndev]).transpose(0, 1)
            down_qzeros = down_qzeros.contiguous()
            ffn_down_qzeros_list.append(down_qzeros)

            down_g_idx = state_dict[naming.down_g_idx(i)]
            if transpose_weight:
                down_g_idx = down_g_idx.reshape([d, ndev, di // ndev]).transpose(0, 1)
            down_g_idx = down_g_idx.contiguous()
            ffn_down_g_idx_list.append(down_g_idx)

        # Assign pointers
        for i in range(nlayer):
            self.attn_qkv_qweight[i] = attn_qkv_qweights[i].data_ptr()
            self.attn_qkv_scales[i] = attn_qkv_scales_list[i].data_ptr()
            self.attn_qkv_qzeros[i] = attn_qkv_qzeros_list[i].data_ptr()
            self.attn_qkv_g_idx[i] = attn_qkv_g_idx_list[i].data_ptr()
            self.attn_o_qweight[i] = attn_o_qweights[i].data_ptr()
            self.attn_o_scales[i] = attn_o_scales_list[i].data_ptr()
            self.attn_o_qzeros[i] = attn_o_qzeros_list[i].data_ptr()
            self.attn_o_g_idx[i] = attn_o_g_idx_list[i].data_ptr()
            self.ffn_gate_up_qweight[i] = ffn_gate_up_qweights[i].data_ptr()
            self.ffn_gate_up_scales[i] = ffn_gate_up_scales_list[i].data_ptr()
            self.ffn_gate_up_qzeros[i] = ffn_gate_up_qzeros_list[i].data_ptr()
            self.ffn_gate_up_g_idx[i] = ffn_gate_up_g_idx_list[i].data_ptr()
            self.ffn_down_qweight[i] = ffn_down_qweights[i].data_ptr()
            self.ffn_down_scales[i] = ffn_down_scales_list[i].data_ptr()
            self.ffn_down_qzeros[i] = ffn_down_qzeros_list[i].data_ptr()
            self.ffn_down_g_idx[i] = ffn_down_g_idx_list[i].data_ptr()

    def _init_full_precision_weights(
        self,
        naming,
        state_dict,
        nlayer,
        ndev,
        nh,
        nkvh,
        dh,
        d,
        di,
        torch_dt_mat,
        torch_dt_logits,
        transpose_weight,
        scale_o,
        scale_down,
    ):
        """Initialize full-precision weights"""

        # Full precision implementation (your original code)
        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        def qkv_b_slices(_i):
            _QB = (
                state_dict[naming.attn_q_b(_i)]
                .reshape([nh, 2, dh // 2])
                .transpose(1, 2)
            )
            _KB = (
                state_dict[naming.attn_k_b(_i)]
                .reshape([nkvh, 2, dh // 2])
                .transpose(1, 2)
            )
            _VB = state_dict[naming.attn_v_b(_i)].reshape([nkvh, dh // 2, 2])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_QB[_idev * _nh : (_idev + 1) * _nh, :, :].flatten())
                _result.append(_KB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
                _result.append(_VB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
            return _result

        if naming.attn_q_b(0) in state_dict:
            self.qkv_b_tensors = [
                torch.concat(qkv_b_slices(i)).to(torch_dt_logits) for i in range(nlayer)
            ]
            self.qkv_b_tensor_ptrs = [
                self.qkv_b_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_qkv_b = (c_void_p * nlayer)(*self.qkv_b_tensor_ptrs)
        else:
            self.attn_qkv_b = None

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_o
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = (
                    self.gate_up_tensors[i]
                    .reshape(ndev, 2 * di // ndev, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        self.ffn_down_tensor = [
            (
                state_dict[naming.down(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, di // ndev])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.down(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_down
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)


class JiugeBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


class JiugeForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_

        print("Loading model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
            self.model_type = config["model_type"]

        # Check if model is quantized
        is_quantized = "quantization_config" in config
        quantization_config = config.get("quantization_config", {})

        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        transpose_weight = (
            device != DeviceType.DEVICE_TYPE_ASCEND
        )  # y = xW is faster than y=xW^T on Ascend

        if "llama" == config["model_type"]:
            model = (
                transformers.LlamaForCausalLM.from_pretrained(model_dir_path)
                .cpu()
                .half()
            )
            self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
            self.weights = JiugeWeightsImpl(
                self.meta,
                LlamaWeightsNaming(),
                model.state_dict(),
                ndev=ndev,
                transpose_weight=transpose_weight,
                is_quantized=is_quantized,
                quantization_config=quantization_config,
            )
        elif "fm9g" == config["model_type"] or "minicpm" == config["model_type"]:
            if any(
                file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()
            ):
                state_dict = load_all_safetensors_from_dir(model_dir_path)
            else:
                state_dict = torch.load(
                    os.path.join(model_dir_path, "pytorch_model.bin"),
                    weights_only=True,
                    map_location="cpu",
                )
            naming = GPTQWeightsNaming() if is_quantized else LlamaWeightsNaming()
            if naming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    naming,
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    is_quantized=is_quantized,
                    quantization_config=quantization_config,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "fm9g7b" == config["model_type"]:
            if any(
                file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()
            ):
                state_dict = load_all_safetensors_from_dir(model_dir_path)
            else:
                state_dict = torch.load(
                    os.path.join(model_dir_path, "pytorch_model.bin"),
                    weights_only=True,
                    map_location="cpu",
                )
            naming = GPTQWeightsNaming() if is_quantized else LlamaWeightsNaming()
            if naming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    naming,
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    is_quantized=is_quantized,
                    quantization_config=quantization_config,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "qwen2" == config["model_type"]:
            state_dict = load_all_safetensors_from_dir(model_dir_path)
            naming = GPTQWeightsNaming() if is_quantized else LlamaWeightsNaming()
            if naming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config, max_tokens=max_tokens)
                self.weights = JiugeWeightsImpl(
                    self.meta,
                    naming,
                    state_dict,
                    ndev=ndev,
                    transpose_weight=transpose_weight,
                    is_quantized=is_quantized,
                    quantization_config=quantization_config,
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path
                )
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_jiuge_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = JiugeBatchedTask(tasks)
        infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            if self.model_type == "qwen2":
                token_piece = self.tokenizer.convert_ids_to_tokens(output_tokens[0])
                output_str = (
                    piece_to_text(token_piece).replace("▁", " ").replace("<0x0A>", "\n")
                )
            else:
                output_str = (
                    self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                    .replace("▁", " ")
                    .replace("<0x0A>", "\n")
                )
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
        tasks = [
            InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id)
            for i in range(batch_size)
        ]
        kv_caches = [KVCache(self) for _ in range(batch_size)]

        nll = 0.0
        total_len = 0

        for i in range(0, len(test_sequences), batch_size):
            batch_id = 0
            true_tokens = []
            while batch_id < batch_size and batch_id + i < len(test_sequences):
                input_tokens = test_sequences[i + batch_id][:-1]
                true_tokens.extend(test_sequences[i + batch_id][1:])
                tasks[batch_id].tokens = input_tokens
                tasks[batch_id].bind_kvcache(kv_caches[batch_id])
                batch_id += 1

            batch_inputs = JiugeBatchedTask(tasks[:batch_id])
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            forward_batch(
                self.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.kv_caches,
                logits.data_ptr(),
            )

            logits = logits.float()
            token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (ntok, vocab)
            token_logprobs = log_probs[
                torch.arange(batch_inputs.ntok), token_ids
            ]  # (ntok,)

            start = 0
            for l in batch_inputs.req_lens_list:
                nll += -token_logprobs[start : start + l].sum().item()
                start += l
            total_len += token_logprobs.numel()

        for task in tasks:
            task.release_kvcache()

        return math.exp(nll / total_len)

    def destroy_model_instance(self):
        destroy_jiuge_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    else:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = JiugeForCauslLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
