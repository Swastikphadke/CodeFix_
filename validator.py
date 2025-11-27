"""
Validator for KV-Cached Multi-Head Attention - FINAL VERSION

Supports:
- visible test_cases.json (1 basic test)
- hidden test_cases_hidden.json (10 tests)
Understands special placeholders:
- "_PLACEHOLDER_USE_VISIBLE_TEST_"
- "_TO_BE_GENERATED_"
- "_FROM_TEST_1_OUTPUT_"
"""

import argparse
import json
import sys
import importlib.util
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class AttentionValidator:
    def __init__(self, module_path: str, test_file: str = "kv_attention.py", verbose: bool = False):
        self.module_path = module_path
        self.test_file = test_file
        self.verbose = verbose

        self.module = None
        self.test_cases: List[Dict[str, Any]] = []
        self.tolerance = 1e-4

        # For special hidden test behavior
        self.visible_inputs = None          # loaded from test_cases.json when needed
        self.test1_output: Optional[torch.Tensor] = None  # for _FROM_TEST_1_OUTPUT_

    # ------------------------------------------------------------------
    # Module + test loading
    # ------------------------------------------------------------------
    def load_module(self) -> bool:
        try:
            spec = importlib.util.spec_from_file_location("kv_attention_module", self.module_path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)

            if not hasattr(self.module, "KVCachedMultiHeadAttention"):
                print("✗ Error: Module must define class 'KVCachedMultiHeadAttention'")
                return False

            if self.verbose:
                print("✓ Module loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading module: {e}")
            return False

    def load_test_cases(self) -> bool:
        try:
            with open(self.test_file, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if "test_cases" in data:
                    self.test_cases = data["test_cases"]
                elif "test_case" in data:
                    self.test_cases = [data["test_case"]]
                else:
                    raise ValueError("Invalid test file format: missing 'test_cases' or 'test_case'")
            elif isinstance(data, list):
                self.test_cases = data
            else:
                raise ValueError("Test file must contain dict or list root")

            if self.verbose:
                print(f"✓ Loaded {len(self.test_cases)} test cases")

            return True
        except Exception as e:
            print(f"✗ Error loading test cases: {e}")
            return False

    # ------------------------------------------------------------------
    # Tensor builder that understands placeholders
    # ------------------------------------------------------------------
    def tensor_from_data(self, data: Any) -> torch.Tensor:
        """
        Handles:
        - Normal visible test dicts with numeric 'values'
        - Hidden tests with values:
          * "_PLACEHOLDER_USE_VISIBLE_TEST_"
          * "_TO_BE_GENERATED_"
          * "_FROM_TEST_1_OUTPUT_"
        """
        # Case: dict with 'values'
        if isinstance(data, dict) and "values" in data:
            values = data["values"]
            shape = data.get("shape", None)

            # 1) Visible test or numeric values
            if not isinstance(values, str):
                tensor = torch.tensor(values, dtype=torch.float32)
                if shape is not None:
                    tensor = tensor.view(*shape)
                return tensor

            # 2) Hidden: reuse visible test input tensors
            if values == "_PLACEHOLDER_USE_VISIBLE_TEST_":
                if self.visible_inputs is None:
                    with open("test_cases.json", "r") as f:
                        visible = json.load(f)
                    self.visible_inputs = visible["test_case"]["inputs"]
                base_vals = self.visible_inputs["query"]["values"]
                tensor = torch.tensor(base_vals, dtype=torch.float32)
                if shape is not None:
                    tensor = tensor.view(*shape)
                return tensor

            # 3) Hidden: generate random tensor (seed already set per test)
            if values == "_TO_BE_GENERATED_":
                if shape is None:
                    raise ValueError("Shape must be provided when values == '_TO_BE_GENERATED_'")
                return torch.randn(*shape, dtype=torch.float32)

            # 4) Hidden: use output of Test #1 as cache for Test #2
            if values == "_FROM_TEST_1_OUTPUT_":
                if self.test1_output is None:
                    raise ValueError("Test #1 output not stored yet for '_FROM_TEST_1_OUTPUT_'")
                tensor = self.test1_output
                if shape is not None:
                    tensor = tensor.view(*shape)
                return tensor

            # Unknown pattern
            raise ValueError(f"Unknown 'values' string pattern: {values}")

        # Fallback: try direct tensor conversion
        return torch.tensor(data, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Single test execution
    # ------------------------------------------------------------------
    def run_test_case(self, test_case: Dict[str, Any], idx: int) -> Tuple[bool, str, Optional[Dict]]:
        try:
            config = test_case.get("config", {})
            d_model = config.get("d_model", 64)
            num_heads = config.get("num_heads", 4)
            max_cache_len = config.get("max_cache_len", 2048)
            dropout = config.get("dropout", 0.0)

            # Instantiate model
            ModelCls = self.module.KVCachedMultiHeadAttention
            model = ModelCls(
                d_model=d_model,
                num_heads=num_heads,
                max_cache_len=max_cache_len,
                dropout=dropout
            )
            model.eval()

            # Set seed so _TO_BE_GENERATED_ is deterministic
            seed = test_case.get("seed", 42)
            torch.manual_seed(seed)
            np.random.seed(seed)

            inputs = test_case["inputs"]
            query = self.tensor_from_data(inputs["query"])
            key   = self.tensor_from_data(inputs["key"])
            value = self.tensor_from_data(inputs["value"])

            # Cache handling
            cache_data = inputs.get("cache", None)
            cache = None
            if cache_data is not None:
                cache = {
                    "key":   self.tensor_from_data(cache_data["key"])   if cache_data.get("key") else None,
                    "value": self.tensor_from_data(cache_data["value"]) if cache_data.get("value") else None,
                }

            use_causal_mask = inputs.get("use_causal_mask", True)

            # Forward pass
            with torch.no_grad():
                output, new_cache = model(query, key, value, cache=cache, use_causal_mask=use_causal_mask)

            # If this is test id 1 (hidden tests), store its output for test 2 cache
            test_id_field = test_case.get("id", None)
            if test_id_field == 1:
                self.test1_output = output.detach().clone()

            expected = test_case.get("expected", None)

            # If expected is None -> visible debug test, only runtime + shape checks
            if expected is None:
                return (
                    True,
                    f"✓ Test #{idx} '{test_case.get('name', 'unnamed')}' - Executed successfully (no expected output to validate)",
                    {
                        "output_shape": list(output.shape),
                        "cache_key_shape": list(new_cache["key"].shape) if new_cache.get("key") is not None else None,
                    }
                )

            # ---------- Shape checks for OUTPUT ----------
            exp_out_spec = expected.get("output", None)
            if exp_out_spec is not None:
                exp_shape = exp_out_spec.get("shape", None)
                if exp_shape is not None:
                    if list(output.shape) != exp_shape:
                        return (
                            False,
                            f"✗ Test #{idx} - Output shape mismatch: got {list(output.shape)}, expected {exp_shape}",
                            None
                        )

                # If values are numeric (not placeholder), do numeric comparison
                if not isinstance(exp_out_spec.get("values"), str):
                    expected_output = self.tensor_from_data(exp_out_spec)
                    max_diff = (output - expected_output).abs().max().item()
                    if max_diff > self.tolerance:
                        return (
                            False,
                            f"✗ Test #{idx} - Output values mismatch: max diff = {max_diff:.6f}",
                            {"max_diff": max_diff}
                        )

            # ---------- Shape checks for CACHE ----------
            exp_cache_spec = expected.get("cache", None)
            if exp_cache_spec is not None and new_cache is not None:
                exp_key_spec = exp_cache_spec.get("key", None)
                if exp_key_spec is not None:
                    exp_key_shape = exp_key_spec.get("shape", None)
                    if exp_key_shape is not None and new_cache.get("key") is not None:
                        if list(new_cache["key"].shape) != exp_key_shape:
                            return (
                                False,
                                f"✗ Test #{idx} - Cache key shape mismatch: got {list(new_cache['key'].shape)}, expected {exp_key_shape}",
                                None
                            )
                # We intentionally do NOT compare numeric cache values, as hidden file has placeholders.

            return (
                True,
                f"✓ Test #{idx} '{test_case.get('name', 'unnamed')}' - PASSED (shape + runtime)",
                {
                    "output_shape": list(output.shape),
                    "cache_key_shape": list(new_cache["key"].shape) if new_cache.get("key") is not None else None,
                }
            )

        except Exception as e:
            import traceback
            msg = f"✗ Test #{idx} - Runtime error: {e}"
            if self.verbose:
                msg += "\n" + traceback.format_exc()
            return False, msg, None

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[int, int, List[str]]:
        passed = 0
        total = len(self.test_cases)
        messages: List[str] = []

        for i, tc in enumerate(self.test_cases, 1):
            ok, msg, _ = self.run_test_case(tc, i)
            messages.append(msg)
            if ok:
                passed += 1

        return passed, total, messages

    # ------------------------------------------------------------------
    # Top-level validate
    # ------------------------------------------------------------------
    def validate(self) -> bool:
        print("=" * 70)
        print("KV-Cached Multi-Head Attention - Validator")
        print("=" * 70)

        if not self.load_module():
            return False
        if not self.load_test_cases():
            return False

        print(f"\nRunning {len(self.test_cases)} test case(s)...\n")

        passed, total, messages = self.run_all_tests()

        for m in messages:
            print(m)

        print("\n" + "=" * 70)
        print(f"Results: {passed}/{total} tests passed")
        print("=" * 70)

        if passed == total:
            print("✓ All tests passed!")
            return True
        else:
            print(f"✗ {total - passed} test(s) failed.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate KV-Cached Multi-Head Attention implementation"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="kv_attention.py",
        help="Path to the Python file to validate (default: kv_attention.py)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test_cases.json",
        help="Path to test cases JSON file (default: test_cases.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    validator = AttentionValidator(
        module_path=args.file,
        test_file=args.test_file,
        verbose=args.verbose,
    )
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
