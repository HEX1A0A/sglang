"""CPU regressions for real Engram storage code without importing NPU kernels.

AST extraction isolates runtime imports, not the storage or remapping algorithms.
These tests do not exercise HCCL, graph capture, or the host-offload implementation.
"""

import ast
import copy
import importlib.util
import logging
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import nn

_PYTHON_ROOT = Path(__file__).resolve().parents[4] / "python"
_ENGRAM = "sglang/srt/layers/engram.py"
_MODEL = "sglang/srt/models/deepseek_v4.py"
_MODELSLIM = "sglang/srt/layers/quantization/modelslim/modelslim.py"
_SPEC = {
    "format": "int8_sym",
    "group_size": 32,
    "axis": 1,
    "scale_suffix": "scale",
    "scale_dtype": "float32",
    "output_dtype": "bfloat16",
}


def _nodes(relative_path):
    path = _PYTHON_ROOT / relative_path
    return ast.parse(path.read_text(), filename=str(path)).body


def _execute(nodes, relative_path, namespace):
    module = ast.parse("from __future__ import annotations\n")
    module.body.extend(nodes)
    exec(compile(module, str(_PYTHON_ROOT / relative_path), "exec"), namespace)  # noqa: S102


_ci_spec = importlib.util.spec_from_file_location(
    "_engram_storage_ci", _PYTHON_ROOT / "sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
with patch.dict(sys.modules, {_ci_spec.name: _ci}):
    _ci_spec.loader.exec_module(_ci)
register_npu_ci = _ci.register_npu_ci
register_npu_ci(est_time=5, suite="base-a-test-1-npu-a2")


def _runtime(*, tp=16, rank=0, arch35=False, npu=True, offload=False):
    state = SimpleNamespace(tp=tp, rank=rank, arch35=arch35, npu=npu, offload=offload)
    namespace = {
        "torch": torch,
        "nn": nn,
        "logger": logging.getLogger(__name__),
        "is_npu": lambda: state.npu,
        "is_npu_arch35": lambda: state.arch35,
        "get_tensor_model_parallel_world_size": lambda: state.tp,
        "get_tensor_model_parallel_rank": lambda: state.rank,
        "get_attention_dp_size": lambda: 1,
        "tensor_model_parallel_all_reduce": lambda x: x,
        "envs": SimpleNamespace(
            SGLANG_OPT_ENGRAM_HOST_OFFLOAD=SimpleNamespace(get=lambda: state.offload)
        ),
        "add_prefix": lambda name, prefix: f"{prefix}.{name}" if prefix else name,
        "ReplicatedLinear": lambda *args, **kwargs: nn.Identity(),
    }
    quant_nodes = _nodes("sglang/srt/layers/attention/dsv4/torch_quant.py")
    _execute(
        [
            node
            for node in quant_nodes
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "FP8_BLOCK_SIZE"
                for t in node.targets
            )
        ],
        "sglang/srt/layers/attention/dsv4/torch_quant.py",
        namespace,
    )
    _execute(
        [
            node
            for node in _nodes(_ENGRAM)
            if isinstance(node, ast.ClassDef)
            and node.name in {"EngramEmbedding", "Engram"}
        ],
        _ENGRAM,
        namespace,
    )
    return state, SimpleNamespace(**namespace)


def _modelslim(description):
    model = next(
        n
        for n in _nodes(_MODEL)
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLM"
    )
    remap = next(
        n
        for n in model.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "remap_weight_name_to_dpsk_hf_format"
    )
    namespace = {
        "QuantizationConfig": object,
        "cast": cast,
        "List": list,
        "MappingProxyType": MappingProxyType,
    }
    _execute([remap], _MODEL, namespace)
    source = SimpleNamespace(
        DeepseekV4ForCausalLM=SimpleNamespace(
            remap_weight_name_to_dpsk_hf_format=namespace[remap.name].__func__
        )
    )
    _execute(
        [
            n
            for n in _nodes(_MODELSLIM)
            if isinstance(n, ast.ClassDef) and n.name == "ModelSlimConfig"
        ],
        _MODELSLIM,
        namespace,
    )
    with patch.dict(sys.modules, {"sglang.srt.models.deepseek_v4": source}):
        return namespace["ModelSlimConfig"](description)


def _source(rows, dim=64):
    codes = (torch.arange(rows * dim).reshape(rows, dim) % 255 - 127).to(torch.int8)
    scales = (torch.arange(rows * (dim // 32)).reshape(rows, -1) % 9 + 1).float() / 64
    return codes, scales


@pytest.mark.parametrize(
    "rows,physical,valid",
    [(384006168, 24000386, 24000378), (384016682, 24001043, 24001037)],
)
def test_real_checkpoint_tp16_partition_without_allocating_table(rows, physical, valid):
    _, runtime = _runtime(rank=15)
    with torch.device("meta"):
        table = runtime.EngramEmbedding(rows, 256, storage_spec=_SPEC)
    assert (table.rows, table.valid_rows, table.row_start) == (
        physical,
        valid,
        15 * physical,
    )
    assert table.weight.shape == (physical, 256)
    assert table.scale.shape == (physical, 8)


@pytest.mark.parametrize("rows", [248, 250])
@pytest.mark.parametrize("arch35", [False, True])
def test_int8_tp16_shards_reconstruct_reference_and_mask_padding(rows, arch35):
    state, runtime = _runtime(arch35=arch35)
    codes, scales = _source(rows)
    ids = torch.tensor([[0, 15, 16, rows - 1], [-1, rows, 31, 32]])
    expected = (codes.float() * scales.repeat_interleave(32, dim=-1)).bfloat16()
    expected = expected[ids.clamp(0, rows - 1)].masked_fill(
        ((ids < 0) | (ids >= rows)).unsqueeze(-1), 0
    )
    outputs = []
    for rank in range(16):
        state.rank = rank
        table = runtime.EngramEmbedding(rows, 64, storage_spec=_SPEC)
        table.weight.weight_loader(table.weight, codes)
        table.scale.weight_loader(table.scale, scales)
        table.validate_loaded_weights()
        outputs.append(table._lookup(ids))
        assert table._lookup(ids[:0]).shape == (0, 4, 64)
        assert torch.count_nonzero(table.weight[table.valid_rows :]) == 0
        assert torch.count_nonzero(table.scale[table.valid_rows :]) == 0
    torch.testing.assert_close(
        torch.stack(outputs).float().sum(0).bfloat16(), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("arch35", [False, True])
def test_native_fallback_preserves_storage_and_lookup(arch35):
    _, runtime = _runtime(tp=2, rank=1, arch35=arch35)
    table = runtime.EngramEmbedding(5, 64)
    source = (torch.arange(5 * 64).reshape(5, 64) % 5 - 2).float()
    if arch35:
        assert table.weight.dtype == torch.float8_e4m3fn
        assert table.scale.dtype == torch.float8_e8m0fnu
        scale_bits = (
            torch.tensor([[126, 128]], dtype=torch.uint8).expand(5, 2).contiguous()
        )
        table.weight.weight_loader(table.weight, source.to(torch.float8_e4m3fn))
        table.scale.weight_loader(table.scale, scale_bits.view(torch.float8_e8m0fnu))
        expected = (source * torch.tensor([0.5, 2.0]).repeat_interleave(32)).bfloat16()
        assert (
            torch.count_nonzero(table.scale[table.valid_rows :].view(torch.uint8)) == 0
        )
    else:
        assert table.weight.dtype == torch.bfloat16
        assert "scale" not in dict(table.named_parameters())
        table.weight.weight_loader(table.weight, source.bfloat16())
        expected = source.bfloat16()
    ids = torch.tensor([[2, 3, 4, 5]])
    expected = (
        torch.stack(
            [torch.zeros(64), expected[3].float(), expected[4].float(), torch.zeros(64)]
        )
        .bfloat16()
        .unsqueeze(0)
    )
    torch.testing.assert_close(table._lookup(ids), expected, rtol=0, atol=0)
    table.validate_loaded_weights()


def test_empty_forward_and_empty_rank():
    _, runtime = _runtime(tp=16, rank=15)
    table = runtime.EngramEmbedding(3, 64, storage_spec=_SPEC)
    codes, scales = _source(3)
    table.weight.weight_loader(table.weight, codes)
    table.scale.weight_loader(table.scale, scales)
    assert table.valid_rows == 0
    assert torch.count_nonzero(table._lookup(torch.tensor([[0, 2, 3]]))) == 0
    assert table(torch.empty((0, 3), dtype=torch.long)).shape == (0, 3, 64)


@pytest.mark.parametrize(
    "parameter,wrong",
    [("weight", "dtype"), ("scale", "dtype"), ("weight", "rows"), ("scale", "columns")],
)
def test_loader_rejects_wrong_dtype_and_shape(parameter, wrong):
    _, runtime = _runtime(tp=1)
    table = runtime.EngramEmbedding(3, 64, storage_spec=_SPEC)
    codes, scales = _source(3)
    tensor = codes if parameter == "weight" else scales
    if wrong == "dtype":
        tensor = tensor.bfloat16()
    elif wrong == "rows":
        tensor = tensor[:-1]
    else:
        tensor = tensor[:, :-1]
    param = getattr(table, parameter)
    with pytest.raises(ValueError, match="INT8 Engram expects"):
        param.weight_loader(param, tensor)


def test_missing_scale_fails_post_load_validation():
    _, runtime = _runtime(tp=1)
    table = runtime.EngramEmbedding(3, 64, storage_spec=_SPEC)
    codes, _ = _source(3)
    table.weight.weight_loader(table.weight, codes)
    with pytest.raises(ValueError, match="Missing.*scale"):
        table.validate_loaded_weights()


def test_int8_offload_is_rejected_before_manager_allocation():
    _, runtime = _runtime(offload=True)
    with pytest.raises(NotImplementedError, match="device-resident"):
        runtime.EngramEmbedding(
            3, 64, layer_hash_index=0, config=SimpleNamespace(), storage_spec=_SPEC
        )


@pytest.mark.parametrize(
    "change",
    [
        {"format": "unknown"},
        {"group_size": 64},
        {"axis": 0},
        {"scale_dtype": "bfloat16"},
        {"output_dtype": "float32"},
        {"scale_suffix": "weight_scale_inv"},
    ],
)
def test_unsupported_explicit_metadata_is_not_silently_native(change):
    _, runtime = _runtime()
    with pytest.raises(ValueError, match="Unsupported"):
        runtime.EngramEmbedding(3, 64, storage_spec={**_SPEC, **change})


@pytest.mark.parametrize("canonical", [False, True])
def test_nested_metadata_remap_and_engram_parent_wiring(canonical):
    prefix = "model." if canonical else ""
    modules = {f"{prefix}layers.{layer}.engram.embed": dict(_SPEC) for layer in (1, 14)}
    description = {
        "hc_head_base": "FLOAT",
        "layers.1.engram.embed.weight": "FLOAT",
        "optional": {
            "unrelated": True,
            "embedding_storage": {"version": 1, "modules": modules},
        },
    }
    original = copy.deepcopy(description)
    quant = _modelslim(description)
    assert description == original
    normalized = quant.quant_description["optional"]
    assert normalized["unrelated"] is True
    assert set(normalized["embedding_storage"]["modules"]) == {
        "model.layers.1.engram.embed",
        "model.layers.14.engram.embed",
    }
    assert quant.quant_description["model.layers.1.engram.embed.weight"] == "FLOAT"
    _, runtime = _runtime()
    config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=4, hc_mult=2)
    layout = SimpleNamespace(
        layer_ids=[1, 14],
        num_embeddings=[32, 34],
        head_dim=64,
        max_ngram_size=2,
        n_heads=2,
    )
    for layer in (1, 14):
        model = runtime.Engram(
            config,
            layer,
            layout,
            quant_config=quant,
            prefix=f"model.layers.{layer}.engram",
        )
        assert model.embed.weight.dtype == torch.int8
        assert model.embed.scale.dtype == torch.float32


@pytest.mark.parametrize("arch35", [False, True])
def test_modelslim_without_storage_metadata_keeps_native_format(arch35):
    quant = _modelslim({"model.layers.1.engram.embed.weight": "FLOAT"})
    _, runtime = _runtime(arch35=arch35)
    config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=4, hc_mult=2)
    layout = SimpleNamespace(
        layer_ids=[1], num_embeddings=[32], head_dim=64, max_ngram_size=2, n_heads=2
    )
    model = runtime.Engram(
        config, 1, layout, quant_config=quant, prefix="model.layers.1.engram"
    )
    assert model.embed.weight.dtype == (
        torch.float8_e4m3fn if arch35 else torch.bfloat16
    )
    if arch35:
        assert model.embed.scale.dtype == torch.float8_e8m0fnu
    else:
        assert "scale" not in dict(model.embed.named_parameters())


def test_parent_rejects_unknown_storage_version():
    quant = SimpleNamespace(
        quant_description={
            "optional": {
                "embedding_storage": {
                    "version": 2,
                    "modules": {"model.layers.1.engram.embed": dict(_SPEC)},
                }
            }
        }
    )
    _, runtime = _runtime()
    config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=4, hc_mult=2)
    layout = SimpleNamespace(layer_ids=[1])
    with pytest.raises(ValueError, match="storage version"):
        runtime.Engram(
            config, 1, layout, quant_config=quant, prefix="model.layers.1.engram"
        )


def test_non_npu_keeps_existing_divisibility_requirement():
    _, runtime = _runtime(npu=False)
    with pytest.raises(AssertionError):
        runtime.EngramEmbedding(17, 64)
    with pytest.raises(ValueError, match="Unsupported"):
        runtime.EngramEmbedding(32, 64, storage_spec=_SPEC)
