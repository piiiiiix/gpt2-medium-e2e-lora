"""Run JSON file-lock regressions without model downloads: python src/verify_file_io.py."""

import ctypes
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from utils.runs import write_json


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / "progress.json"
        write_json(self.target, {"step": 0})

    def test_transient_lock(self):
        replace = os.replace
        with (
            patch(
                "utils.runs.os.replace", side_effect=[PermissionError(), PermissionError(), None]
            ) as mocked,
            patch("utils.runs.time.sleep"),
        ):
            # Perform the real replacement on the last attempt.
            def attempt(src, dst):
                if mocked.call_count < 3:
                    raise PermissionError("sharing lock")
                replace(src, dst)

            mocked.side_effect = attempt
            write_json(self.target, {"step": 1})
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(json.loads(self.target.read_text()), {"step": 1})
        self.assertEqual(list(self.target.parent.glob("*.tmp")), [])

    def test_persistent_lock_preserves_previous_json(self):
        with (
            patch("utils.runs.os.replace", side_effect=PermissionError("locked")) as mocked,
            patch("utils.runs.time.sleep"),
            self.assertRaises(PermissionError),
        ):
            write_json(self.target, {"step": 1})
        self.assertEqual(mocked.call_count, 8)
        self.assertEqual(json.loads(self.target.read_text()), {"step": 0})
        self.assertEqual(list(self.target.parent.glob("*.tmp")), [])

    def test_concurrent_writers(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda step: write_json(self.target, {"step": step}), range(40)))
        self.assertIn(json.loads(self.target.read_text())["step"], range(40))
        self.assertEqual(list(self.target.parent.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows sharing semantics")
    def test_real_windows_read_handle(self):
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Read sharing without FILE_SHARE_DELETE reproduces the failing replace.
        handle = kernel.CreateFileW(str(self.target), 0x80000000, 1, None, 3, 0, None)
        self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
        closed = False

        def release(_delay):
            nonlocal closed
            if not closed:
                self.assertTrue(kernel.CloseHandle(handle))
                closed = True

        try:
            with patch("utils.runs.time.sleep", side_effect=release) as sleep:
                write_json(self.target, {"step": 1})
            self.assertTrue(sleep.called)
            self.assertEqual(json.loads(self.target.read_text()), {"step": 1})
        finally:
            if not closed:
                kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
