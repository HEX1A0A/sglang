# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for ModelSlim quantization metadata source selection.

AST extraction runs the actual loader functions with the actual checkpoint
metadata resolver, without importing the GPU/NPU runtime. These tests validate
source selection, not ModelSlim kernel initialization or full runtime imports.
"""

import ast
import copy
import glob
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

_PYTHON_ROOT = Path(__file__).resolve().parents[4] / "python"


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _PYTHON_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "_modelslim_quant_ci", "sglang/test/ci/ci_register.py"
).register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_weight_utils():
    resolver = _load_module(
        "_modelslim_checkpoint_quantization",
        "sglang/srt/model_loader/checkpoint_quantization.py",
    )
    module = ModuleType("_modelslim_weight_utils")
    module.__dict__.update(
        Path=Path,
        os=os,
        glob=glob,
        json=json,
        resolve_checkpoint_quant_spec=resolver.resolve_checkpoint_quant_spec,
        REQUANTIZATION_METHODS=set(),
        get_quantization_config=lambda name: None,
        get_lock=lambda *args: nullcontext(),
        snapshot_download=lambda *args, **kwargs: None,
        huggingface_hub=SimpleNamespace(constants=SimpleNamespace(HF_HUB_OFFLINE=True)),
        DisabledTqdm=object,
    )
    path = _PYTHON_ROOT / "sglang/srt/model_loader/weight_utils.py"
    source = ast.parse(path.read_text(), filename=str(path))
    function_names = {
        "get_quant_config",
        "_modelopt_quant_section",
        "_resolve_explicit_draft_quant_config",
    }
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in functions} == function_names
    tree = ast.parse("from __future__ import annotations\n")
    tree.body.extend(functions)
    exec(  # noqa: S102 - actual source functions, isolated from runtime imports
        compile(tree, str(path), "exec"), module.__dict__
    )
    return module


weight_utils = _load_weight_utils()


class TestModelSlimQuantConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_path = self.temp_dir.name
        self.description = {
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
        }
        self.summary = {
            "quant_method": "ascend",
            "model_quant_type": "W8A8_DYNAMIC",
        }
        self.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
        self.quant_cls = MagicMock()
        self.quant_cls.get_config_filenames.return_value = [
            "quant_model_description.json"
        ]
        self.quant_cls.from_config.side_effect = lambda config: config
        self.quant_cls_patch = patch.object(
            weight_utils, "get_quantization_config", return_value=self.quant_cls
        )
        self.quant_cls_patch.start()
        self.addCleanup(self.quant_cls_patch.stop)

    def write_sidecar(self):
        Path(self.model_path, "quant_model_description.json").write_text(
            json.dumps(self.description)
        )

    def load(self, inline, *, quantization="modelslim", model_path=None):
        hf_config = {"quantization_config": inline}
        model_config = SimpleNamespace(
            quantization=quantization,
            model_path=model_path or self.model_path,
            hf_config=hf_config,
            is_draft_model=False,
            revision=None,
        )
        load_config = SimpleNamespace(model_loader_extra_config=None, download_dir=None)
        return weight_utils.get_quant_config(
            model_config, load_config, self.packed_modules_mapping
        )

    def assert_sidecar_loaded(self, result):
        for key, value in self.description.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["packed_modules_mapping"], self.packed_modules_mapping)
        self.assertNotIn("model_quant_type", result)
        self.quant_cls.from_config.assert_called_once()

    def test_local_sidecar_takes_precedence_over_hf_summary(self):
        self.write_sidecar()
        original = copy.deepcopy(self.summary)
        self.assert_sidecar_loaded(self.load(self.summary))
        self.assertEqual(self.summary, original)

    def test_local_sidecar_takes_precedence_over_inline_description(self):
        self.write_sidecar()
        inline = {key: "FLOAT" for key in self.description}
        inline["quant_method"] = "modelslim"
        original = copy.deepcopy(inline)
        self.assert_sidecar_loaded(self.load(inline))
        self.assertEqual(inline, original)

    def test_inline_only_description_is_preserved_without_mutating_source(self):
        inline = {"quant_method": "modelslim", **self.description}
        original = copy.deepcopy(inline)
        result = self.load(inline)
        for key, value in inline.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["packed_modules_mapping"], self.packed_modules_mapping)
        self.assertEqual(inline, original)
        self.quant_cls.get_config_filenames.assert_not_called()

    def test_sidecar_without_inline_metadata_keeps_file_loading_path(self):
        self.write_sidecar()
        self.assert_sidecar_loaded(self.load(None))

    def test_non_modelslim_inline_config_is_unchanged(self):
        self.write_sidecar()
        inline = {"quant_method": "fp8", "activation_scheme": "dynamic"}
        original = copy.deepcopy(inline)
        result = self.load(inline, quantization="fp8")
        self.assertEqual(result["quant_method"], "fp8")
        self.assertEqual(result["activation_scheme"], "dynamic")
        self.assertNotIn(next(iter(self.description)), result)
        self.assertEqual(inline, original)
        self.quant_cls.get_config_filenames.assert_not_called()

    def test_remote_summary_downloads_sidecar(self):
        self.write_sidecar()
        with (
            patch.object(weight_utils, "get_lock"),
            patch.object(
                weight_utils, "snapshot_download", return_value=self.model_path
            ) as download,
        ):
            result = self.load(self.summary, model_path="org/modelslim-checkpoint")
        self.assert_sidecar_loaded(result)
        download.assert_called_once()
        self.assertEqual(download.call_args.args[0], "org/modelslim-checkpoint")
        self.assertEqual(download.call_args.kwargs["allow_patterns"], "*.json")

    def test_summary_without_sidecar_reports_missing_config(self):
        with self.assertRaisesRegex(ValueError, "Cannot find the config file"):
            self.load(self.summary)
        self.quant_cls.from_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
