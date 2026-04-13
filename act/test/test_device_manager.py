#!/usr/bin/env python3
"""Device manager tests focused on real CPU/MPS behavior."""

import argparse
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


if __name__ == "__main__":
    unittest.main()
