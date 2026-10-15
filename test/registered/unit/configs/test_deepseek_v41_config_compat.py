"""V4.1 config compatibility without importing GPU/NPU runtime modules.

Load the actual config aliases, helpers, and HF parser via AST, using real
Transformers configs and local JSON fixtures. This covers parser ordering and
registry reloads, not model execution or the complete SGLang import graph.
"""

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from transformers import AutoConfig, DeepseekV3Config, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

_ROOT = Path(__file__).resolve().parents[4]
_PYTHON = _ROOT / "python"
_HF_UTILS = _PYTHON / "sglang/srt/utils/hf_transformers"


def _load_ci():
    spec = importlib.util.spec_from_file_location(
        "_dsv41_config_ci", _PYTHON / "sglang/test/ci/ci_register.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
    return module.register_cpu_ci


register_cpu_ci = _load_ci()
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_config_namespace():
    namespace = {
        "_HFDeepseekV3Config": DeepseekV3Config,
        "PretrainedConfig": PretrainedConfig,
        "AutoConfig": AutoConfig,
        "_CONFIG_REGISTRY": {},
        "Optional": Optional,
        "ModelConfigParserBase": object,
        "register_model_config_parser": lambda name: lambda cls: cls,
        "_try_load_longcat_config": lambda *args, **kwargs: None,
        "_is_deepseek_ocr_model": lambda config: False,
        "_is_deepseek_ocr2_model": lambda config: False,
        "normalize_rope_scaling_compat": lambda config: None,
    }
    class_names = {
        "_DeepseekV4ConfigAlias",
        "_DeepseekV41ConfigAlias",
        "_DeepseekV41PublishedConfigAlias",
        "_DeepseekV41TextConfigAlias",
    }
    function_names = {
        "_normalize_deepseek_v41_config",
        "get_hf_text_config",
        "_patch_text_config",
    }
    common = ast.parse((_HF_UTILS / "common.py").read_text())
    nodes = [
        node
        for node in ast.walk(common)
        if (isinstance(node, ast.ClassDef) and node.name in class_names)
        or (isinstance(node, ast.FunctionDef) and node.name in function_names)
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Name)
            and node.value.id in class_names
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "_CONFIG_REGISTRY"
        )
    ]
    nodes.sort(key=lambda node: node.lineno)
    exec(  # noqa: S102 - execute only selected definitions from this checkout
        compile(ast.Module(body=nodes, type_ignores=[]), "common.py", "exec"),
        namespace,
    )
    config = ast.parse((_HF_UTILS / "config.py").read_text())
    nodes = [
        node
        for node in config.body
        if isinstance(node, ast.ClassDef) and node.name == "HfModelConfigParser"
    ]
    exec(  # noqa: S102 - execute the real parser without the runtime import graph
        compile(ast.Module(body=nodes, type_ignores=[]), "config.py", "exec"),
        namespace,
    )
    return namespace


_NAMESPACE = _load_config_namespace()
_normalize = _NAMESPACE["_normalize_deepseek_v41_config"]
_V41 = _NAMESPACE["_DeepseekV41ConfigAlias"]
_PublishedV41 = _NAMESPACE["_DeepseekV41PublishedConfigAlias"]
_TextV41 = _NAMESPACE["_DeepseekV41TextConfigAlias"]

_FIELDS = {
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "engram_pad_token_id": 2,
}
_ALIASES = {
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
}


class TestDeepseekV41ConfigCompat(unittest.TestCase):
    def setUp(self):
        # Do not leave our isolated config classes in Transformers' global
        # registry when this module runs alongside other SGLang tests.
        mapping_patch = patch.dict(CONFIG_MAPPING._extra_content)
        mapping_patch.start()
        self.addCleanup(mapping_patch.stop)
        for model_type, config_cls in _NAMESPACE["_CONFIG_REGISTRY"].items():
            AutoConfig.register(model_type, config_cls, exist_ok=True)

    def _parse(self, contents):
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text(json.dumps(contents))
            return _NAMESPACE["HfModelConfigParser"]().parse(
                model_dir, trust_remote_code=False
            )

    def _assert_fields(self, config):
        for published, legacy in _ALIASES.items():
            with self.subTest(field=published):
                self.assertEqual(getattr(config, published), _FIELDS[published])
                self.assertEqual(getattr(config, legacy), _FIELDS[published])

    def test_published_nested_config_survives_registry_reload(self):
        config = self._parse(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "vision_n_layers": 27,
                "text_config": {
                    "model_type": "deepseek_v41_text",
                    "hidden_size": 5120,
                    "num_attention_heads": 64,
                    "compress_ratios": [0, 0, 1, 1, 2, 2],
                    **_FIELDS,
                },
            }
        )
        self.assertEqual(config.model_type, "deepseek_v4.1")
        self.assertEqual(config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.vision_n_layers, 27)
        self.assertEqual(config.compress_ratios, [0, 0, 1, 1, 2, 2])
        self._assert_fields(config)
        text = _NAMESPACE["get_hf_text_config"](config)
        self.assertEqual(text.model_type, "deepseek_v4.1")
        self.assertEqual(text.hidden_size, 5120)
        self._assert_fields(text)

    def test_flat_published_text_config(self):
        config = self._parse(
            {
                "model_type": "deepseek_v41_text",
                "architectures": ["DeepseekV41ForCausalLM"],
                "hidden_size": 5120,
                **_FIELDS,
            }
        )
        self.assertEqual(config.model_type, "deepseek_v4.1")
        self.assertEqual(config.architectures, ["DeepseekV4ForCausalLM"])
        self.assertTrue(config.hc_pre_from_prev_sublayer)
        self.assertFalse(config.q_head_norm)
        self._assert_fields(config)

    def test_native_v41_values_and_custom_architecture_are_preserved(self):
        fields = {legacy: _FIELDS[new] for new, legacy in _ALIASES.items()}
        config = self._parse(
            {
                "model_type": "deepseek_v4.1",
                "architectures": ["DeepseekV4ForCausalLM", "CustomDrafter"],
                "hidden_size": 5120,
                **fields,
            }
        )
        self.assertEqual(
            config.architectures, ["DeepseekV4ForCausalLM", "CustomDrafter"]
        )
        for legacy, value in fields.items():
            self.assertEqual(getattr(config, legacy), value)
        self.assertFalse(hasattr(config, "kv_source_layer_ids"))

    def test_explicit_legacy_value_wins_but_inherited_default_does_not(self):
        config = _V41(
            engram_pad_token_id=9,
            candidate_source_layer_id=20,
            candidate_source_layer=14,
        )
        self.assertEqual(config.engram_pad_id, 2)
        _normalize(config)
        self.assertEqual(config.engram_pad_id, 9)
        self.assertEqual(config.engram_pad_token_id, 9)
        self.assertEqual(config.candidate_source_layer, 14)
        self.assertEqual(config.candidate_source_layer_id, 20)

    def test_object_text_config_overrides_outer_defaults(self):
        text = _TextV41(hidden_size=5120, **_FIELDS)
        config = _PublishedV41(
            architectures=["DeepseekV41ForCausalLM"], text_config=text
        )
        namespace = _NAMESPACE["HfModelConfigParser"].parse.__globals__
        with (
            patch.dict(namespace, {"_try_load_longcat_config": lambda *a, **k: config}),
            patch.object(_PublishedV41, "from_pretrained", return_value=config),
        ):
            parsed = _NAMESPACE["HfModelConfigParser"]().parse(
                "unused-local-path", trust_remote_code=False
            )
        self.assertEqual(parsed.hidden_size, 5120)
        self.assertEqual(text.model_type, "deepseek_v4.1")
        self._assert_fields(parsed)
        self._assert_fields(text)

    def test_v4_and_unrelated_models_are_unchanged(self):
        for model_type in ("deepseek_v4", "deepseek_v3", "unrelated"):
            with self.subTest(model_type=model_type):
                config = PretrainedConfig(
                    model_type=model_type,
                    architectures=["DeepseekV41ForCausalLM"],
                    **_FIELDS,
                )
                before = copy.deepcopy(vars(config))
                _normalize(config)
                self.assertEqual(vars(config), before)

    def test_missing_source_fields_are_not_synthesized(self):
        config = _V41()
        _normalize(config)
        self.assertFalse(hasattr(config, "kv_source_layers"))
        self.assertFalse(hasattr(config, "index_source_layers"))
        self.assertFalse(hasattr(config, "candidate_source_layer"))


if __name__ == "__main__":
    unittest.main()
