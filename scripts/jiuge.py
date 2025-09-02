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
                # self._init_quantized_weights_i8(
                self._init_quantized_weights_i8_naive(
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
    
    def _init_quantized_weights_i8_naive(
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
        """Initialize quantized weights by dequantizing them to full precision"""
        # Initialize all tensors as empty lists
        self.attn_q_tensors = []
        self.attn_k_tensors = []
        self.attn_v_tensors = []
        self.attn_o_tensors = []
        self.gate_tensors = []
        self.up_tensors = []
        self.down_tensors = []
        
        # Check if bias terms exist and initialize them if they do
        self.has_bias = naming.attn_q_b(0) in state_dict
        if self.has_bias:
            self.attn_q_b_tensors = []
            self.attn_k_b_tensors = []
            self.attn_v_b_tensors = []
        
        # First, make sure all norm tensors are initialized
        torch_dt_norm = self.output_norm_tensor.dtype
        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        print(f"nh: {nh}, nkvh: {nkvh}, dh: {dh}, d: {d}")
        print(f"qkv qw input shapes - Q qw:{state_dict[naming.attn_q_qweight(0)].shape}, K qw:{state_dict[naming.attn_k_qweight(0)].shape}, V qw:{state_dict[naming.attn_v_qweight(0)].shape}")
        print(f"qkv bias input shapes - Qb:{state_dict[naming.attn_q_b(0)].shape}, Kb:{state_dict[naming.attn_k_b(0)].shape}, Vb:{state_dict[naming.attn_v_b(0)].shape}")
        print(f"down input shapes - o qw:{state_dict[naming.attn_o_qweight(0)].shape}")
        print(f"gate up input shapes - gate qw:{state_dict[naming.gate_qweight(0)].shape}, up qw:{state_dict[naming.up_qweight(0)].shape}")
        print(f"down input shapes - down qw:{state_dict[naming.down_qweight(0)].shape}")
        
        # Process each layer
        for i in range(nlayer):
            
            # Dequantize Q weights
            q_dequantized = self._dequantize_weight(
                state_dict[naming.attn_q_qweight(i)],
                state_dict[naming.attn_q_scales(i)],
                state_dict[naming.attn_q_qzeros(i)],
                state_dict[naming.attn_q_g_idx(i)]
            )
            self.attn_q_tensors.append(q_dequantized)
            
            # Dequantize K weights
            k_dequantized = self._dequantize_weight(
                state_dict[naming.attn_k_qweight(i)],
                state_dict[naming.attn_k_scales(i)],
                state_dict[naming.attn_k_qzeros(i)],
                state_dict[naming.attn_k_g_idx(i)]
            )
            self.attn_k_tensors.append(k_dequantized)
            
            # Dequantize V weights
            v_dequantized = self._dequantize_weight(
                state_dict[naming.attn_v_qweight(i)],
                state_dict[naming.attn_v_scales(i)],
                state_dict[naming.attn_v_qzeros(i)],
                state_dict[naming.attn_v_g_idx(i)]
            )
            self.attn_v_tensors.append(v_dequantized)
            
            # Dequantize O weights
            o_dequantized = self._dequantize_weight(
                state_dict[naming.attn_o_qweight(i)],
                state_dict[naming.attn_o_scales(i)],
                state_dict[naming.attn_o_qzeros(i)],
                state_dict[naming.attn_o_g_idx(i)]
            )
            self.attn_o_tensors.append(o_dequantized)
            
            # Dequantize Gate weights
            gate_dequantized = self._dequantize_weight(
                state_dict[naming.gate_qweight(i)],
                state_dict[naming.gate_scales(i)],
                state_dict[naming.gate_qzeros(i)],
                state_dict[naming.gate_g_idx(i)]
            )
            if not transpose_weight:
                gate_dequantized = gate_dequantized.transpose(0, 1).contiguous()
            self.gate_tensors.append(gate_dequantized)
            
            # Dequantize Up weights
            up_dequantized = self._dequantize_weight(
                state_dict[naming.up_qweight(i)],
                state_dict[naming.up_scales(i)],
                state_dict[naming.up_qzeros(i)],
                state_dict[naming.up_g_idx(i)]
            )
            if not transpose_weight:
                up_dequantized = up_dequantized.transpose(0, 1).contiguous()
            self.up_tensors.append(up_dequantized)
            
            # Dequantize Down weights
            down_dequantized = self._dequantize_weight(
                state_dict[naming.down_qweight(i)],
                state_dict[naming.down_scales(i)],
                state_dict[naming.down_qzeros(i)],
                state_dict[naming.down_g_idx(i)]
            )
            self.down_tensors.append(down_dequantized)
            
            # Handle bias terms if they exist
            if self.has_bias:
                self.attn_q_b_tensors.append(state_dict[naming.attn_q_b(i)])
                self.attn_k_b_tensors.append(state_dict[naming.attn_k_b(i)])
                self.attn_v_b_tensors.append(state_dict[naming.attn_v_b(i)])
        
        # Now create a fake state_dict with dequantized weights
        fake_state_dict = {}
        
        # Add input embedding and output norm (these are not quantized)
        fake_state_dict[naming.input_embd()] = self.input_embd_tensor
        fake_state_dict[naming.output_norm()] = self.output_norm_tensor
        fake_state_dict[naming.output_embd()] = self.output_embd_tensor
        
        # Add layer weights
        for i in range(nlayer):
            fake_state_dict[naming.attn_norm(i)] = self.attn_norm_tensors[i]
            fake_state_dict[naming.attn_q(i)] = self.attn_q_tensors[i]
            fake_state_dict[naming.attn_k(i)] = self.attn_k_tensors[i]
            fake_state_dict[naming.attn_v(i)] = self.attn_v_tensors[i]
            fake_state_dict[naming.attn_o(i)] = self.attn_o_tensors[i]
            fake_state_dict[naming.ffn_norm(i)] = self.ffn_norm_tensors[i]
            fake_state_dict[naming.gate(i)] = self.gate_tensors[i]
            fake_state_dict[naming.up(i)] = self.up_tensors[i]
            fake_state_dict[naming.down(i)] = self.down_tensors[i]
            
            # Add bias terms if they exist
            if self.has_bias:
                fake_state_dict[naming.attn_q_b(i)] = self.attn_q_b_tensors[i]
                fake_state_dict[naming.attn_k_b(i)] = self.attn_k_b_tensors[i]
                fake_state_dict[naming.attn_v_b(i)] = self.attn_v_b_tensors[i]
        
        # Now call the full precision initialization with the dequantized weights
        self._init_full_precision_weights(
            naming,
            fake_state_dict,
            nlayer,
            ndev,
            nh,
            nkvh,
            dh,
            d,
            di,
            self.input_embd_tensor.dtype,  # Use the same dtype as input embedding
            self.input_embd_tensor.dtype,
            transpose_weight,
            scale_o,
            scale_down,
        )

    def _dequantize_weight(self, qweight, scales, qzeros, g_idx):
        """
        Dequantize GPTQ int8 weights to full precision
        
        Args:
            qweight: torch.Tensor - int8 quantized weights (packed format)
            scales: torch.Tensor - float16 scale factors
            qzeros: torch.Tensor - int8 zero points (packed format)
            g_idx: torch.Tensor - group indices mapping weights to scales/zeros
        
        Returns:
            torch.Tensor: Dequantized weights in float16
        """
        # Unpack qweight (int8 weights packed into int32)
        # Each int32 contains 4 int8 weights
        qweight_unpacked = torch.zeros(
            (qweight.shape[0], qweight.shape[1] * 4),
            dtype=torch.int8,
            device=qweight.device
        )
        
        # Unpack each int32 into 4 int8 values
        for i in range(4):
            qweight_unpacked[:, i::4] = (qweight >> (8 * i)) & 0xFF
        
        # Unpack qzeros similarly
        qzeros_unpacked = torch.zeros(
            (qzeros.shape[0], qzeros.shape[1] * 4),
            dtype=torch.int8,
            device=qzeros.device
        )
        for i in range(4):
            qzeros_unpacked[:, i::4] = (qzeros >> (8 * i)) & 0xFF
        
        # Get original weight dimensions
        out_features, in_features = qweight_unpacked.shape
        
        # Reshape scales and zeros to match weight groups
        scales = scales.reshape(-1, 1)
        qzeros_unpacked = qzeros_unpacked.reshape(-1, 1)
        
        # Calculate the number of groups
        group_size = self.group_size
        num_groups = (in_features + group_size - 1) // group_size
        
        # Ensure scales and zeros have the correct shape
        if scales.shape[0] < num_groups:
            scales = scales.repeat(num_groups // scales.shape[0] + 1, 1)
            scales = scales[:num_groups]
        
        if qzeros_unpacked.shape[0] < num_groups:
            qzeros_unpacked = qzeros_unpacked.repeat(num_groups // qzeros_unpacked.shape[0] + 1, 1)
            qzeros_unpacked = qzeros_unpacked[:num_groups]
        
        # Expand scales and zeros to match the weight dimensions
        scales_expanded = torch.zeros((out_features, in_features), 
                                    dtype=scales.dtype, 
                                    device=scales.device)
        zeros_expanded = torch.zeros((out_features, in_features), 
                                dtype=qzeros_unpacked.dtype, 
                                device=qzeros_unpacked.device)
        
        # Apply group-wise scaling
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, in_features)
            
            if start_idx >= in_features:
                break
                
            scales_expanded[:, start_idx:end_idx] = scales[g]
            zeros_expanded[:, start_idx:end_idx] = qzeros_unpacked[g]
        
        # Dequantize weights using: (qweight - zero) * scale
        if self.symmetric:
            dequantized_weight = qweight_unpacked.float() * scales_expanded.float()
        else:
            dequantized_weight = (qweight_unpacked.float() - zeros_expanded.float()) * scales_expanded.float()
        
        # Convert back to float16
        dequantized_weight = dequantized_weight.half()
        
        return dequantized_weight.contiguous()

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
        print(f"nh: {nh}, nkvh: {nkvh}, dh: {dh}, d: {d}")
        print(f"qkv input shapes - Q:{state_dict[naming.attn_q(0)].shape}, K:{state_dict[naming.attn_k(0)].shape}, V:{state_dict[naming.attn_v(0)].shape}")
        print(f"qkv bias input shapes - Qb:{state_dict[naming.attn_q_b(0)].shape}, Kb:{state_dict[naming.attn_k_b(0)].shape}, Vb:{state_dict[naming.attn_v_b(0)].shape}")
        print(f"down input shapes - o:{state_dict[naming.attn_o(0)].shape}")
        print(f"gate up input shapes - gate:{state_dict[naming.gate(0)].shape}, up:{state_dict[naming.up(0)].shape}")
        print(f"down input shapes - down:{state_dict[naming.down(0)].shape}")
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
            # print(f"qkv_b_tensors shapes: {[t.shape for t in self.qkv_b_tensors]}")
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
