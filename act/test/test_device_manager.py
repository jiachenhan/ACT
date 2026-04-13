#!/usr/bin/env python3
"""Device manager tests focused on real CPU/MPS behavior."""

import argparse
from pathlib import Path
import subprocess
import sys
import unittest

import torch

import act.util.device_manager as device_manager
from act.util.cli_utils import add_device_args


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        device_manager._INITIALIZED = False

    def tearDown(self):
        device_manager._INITIALIZED = False

    def test_initialize_cpu(self):
        device_manager.initialize_device(device="cpu", dtype="float32")

        self.assertEqual(str(device_manager.get_default_device()), "cpu")
        self.assertEqual(device_manager.get_default_dtype(), torch.float32)

    def test_initialize_mps(self):
        if not _mps_available():
            self.skipTest("MPS not available")

        device_manager.initialize_device(device="mps", dtype="float32")

        self.assertTrue(str(device_manager.get_default_device()).startswith("mps"))
        self.assertEqual(device_manager.get_default_dtype(), torch.float32)

    def test_mps_float64_fallback_to_float32(self):
        if not _mps_available():
            self.skipTest("MPS not available")

        device_manager.initialize_device(device="mps", dtype="float64")

        self.assertTrue(str(device_manager.get_default_device()).startswith("mps"))
        self.assertEqual(device_manager.get_default_dtype(), torch.float32)

    def test_initialize_apple_alias(self):
        if not _mps_available():
            self.skipTest("MPS not available")

        device_manager.initialize_device(device="apple", dtype="float32")

        self.assertTrue(str(device_manager.get_default_device()).startswith("mps"))
        self.assertEqual(device_manager.get_default_dtype(), torch.float32)

    def test_initialize_cuda(self):
        if not _cuda_available():
            self.skipTest("CUDA not available")

        device_manager.initialize_device(device="cuda", dtype="float32")

        self.assertTrue(str(device_manager.get_default_device()).startswith("cuda"))
        self.assertEqual(device_manager.get_default_dtype(), torch.float32)


class TestCliUtils(unittest.TestCase):
    def test_add_device_args_contains_mps_and_aliases(self):
        parser = argparse.ArgumentParser()
        add_device_args(parser)

        device_action = next(action for action in parser._actions if action.dest == "device")
        self.assertEqual(
            set(device_action.choices or []),
            {"cpu", "cuda", "mps", "gpu", "apple", "metal"},
        )


class TestMpsEndToEnd(unittest.TestCase):
    """Minimal smoke tests that only validate MPS CLI viability."""

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _run_cli(
        cls,
        module_name: str,
        args: list[str],
        timeout: int = 300,
    ) -> subprocess.CompletedProcess:
        """Run a module with fixed MPS + float32 arguments."""
        cmd = [sys.executable, "-m", module_name, *args, "--device", "mps", "--dtype", "float32"]
        return subprocess.run(
            cmd,
            cwd=cls._repo_root(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    @staticmethod
    def _first_downloaded_vnnlib_category() -> str | None:
        """Return one downloaded VNNLIB category for tiny fuzz smoke, if available."""
        from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader

        downloaded = vnnlib_loader.list_downloaded_pairs()
        if not downloaded:
            return None
        return downloaded[0]["category"]

    def _assert_success(
        self,
        result: subprocess.CompletedProcess,
        label: str,
        required_text: str | None = None,
    ) -> None:
        """Assert command success and optionally check a key stdout marker."""
        if result.returncode != 0:
            self.fail(
                f"{label} failed (exit={result.returncode})\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )
        if required_text and required_text not in result.stdout:
            self.fail(
                f"{label} missing expected output: {required_text!r}\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

    def test_front_end_list_with_mps(self):
        """Smoke test: front_end list command runs with MPS flags."""
        if not _mps_available():
            self.skipTest("MPS not available")

        result = self._run_cli("act.front_end", ["--list"])
        self._assert_success(result, "front_end --list", required_text="ACT FRONT-END UNIFIED CATALOG")

    def test_pipeline_verify_act2torch_with_mps(self):
        """Smoke test: pipeline verify act2torch completes on MPS."""
        if not _mps_available():
            self.skipTest("MPS not available")

        result = self._run_cli("act.pipeline", ["--verify", "act2torch"], timeout=600)
        self._assert_success(result, "pipeline --verify act2torch", required_text="VERIFICATION TEST SUMMARY")

    def test_pipeline_fuzz_with_mps_when_data_exists(self):
        """Smoke test: tiny fuzz run on MPS when local VNNLIB data already exists."""
        if not _mps_available():
            self.skipTest("MPS not available")

        category = self._first_downloaded_vnnlib_category()
        if not category:
            self.skipTest("No downloaded VNNLIB category for fuzz smoke")

        result = self._run_cli(
            "act.pipeline",
            [
                "--fuzz",
                "--creator",
                "vnnlib",
                "--category",
                category,
                "--max-instances",
                "1",
                "--iterations",
                "1",
                "--timeout",
                "15",
                "--no-save",
                "--report-interval",
                "1",
            ],
            timeout=600,
        )
        self._assert_success(result, "pipeline --fuzz smoke", required_text="FUZZING COMPLETE")


if __name__ == "__main__":
    unittest.main()
