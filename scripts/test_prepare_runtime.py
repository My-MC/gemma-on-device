"""Validate packaged runtime layouts using hash-pinned synthetic archives."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import prepare_runtime as runtime


class PrepareRuntimeTests(unittest.TestCase):
    def prepare(self, files, system="Windows", machine="AMD64"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as package:
            for name, content in files.items():
                package.writestr(name, content)
        payload = archive.getvalue()
        target = "win32-x64" if system == "Windows" else "linux-x64"
        lock = copy.deepcopy(runtime.LOCK)
        lock["targets"][f"{target}-cuda"]["packages"] = [{
            "url": "https://example.test/runtime.whl",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }]
        with patch.object(runtime, "LOCK", lock), patch.object(
            runtime, "CACHE", root / "cache"
        ), patch.object(runtime, "OUT", root / "out"), patch.object(
            runtime.platform, "system", return_value=system
        ), patch.object(runtime.platform, "machine", return_value=machine), patch.object(
            runtime.sys, "argv", ["prepare_runtime.py", "cuda"]
        ), patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
            runtime.main()
        return root / "out" / target / "cuda"

    def assert_manifest(self, destination, expected):
        manifest = json.loads((destination / "runtime-manifest.json").read_text())
        self.assertEqual(
            manifest["files"],
            {name: hashlib.sha256(content).hexdigest() for name, content in expected.items()},
        )
        for name, content in expected.items():
            self.assertEqual((destination / name).read_bytes(), content)

    def test_windows_moves_only_root_runtime_libraries(self):
        libraries = {
            "onnxruntime.dll": b"ORT core",
            "onnxruntime_providers_cuda.dll": b"CUDA provider",
            "onnxruntime_providers_shared.dll": b"shared provider",
            "onnxruntime_providers_webgpu.dll": b"WebGPU provider",
        }
        retained = {
            "onnxruntime/LICENSE": b"ORT license",
            "onnxruntime_ep_webgpu/LICENSE": b"WebGPU license",
            "onnxruntime_ep_webgpu/dxcompiler.dll": b"DX compiler",
            "onnxruntime_ep_webgpu/dxil.dll": b"DXIL",
            "nvidia/cublas/bin/cublas64_13.dll": b"cuBLAS",
            "nvidia/cudnn/bin/cudnn64_9.dll": b"cuDNN",
            "nvidia/cudnn/LICENSE": b"NVIDIA license",
        }
        nested = {
            f"onnxruntime_ep_webgpu/{name}" if "webgpu" in name else f"onnxruntime/capi/{name}": content
            for name, content in libraries.items()
        }
        destination = self.prepare({
            **nested, **retained,
            "onnxruntime/capi/onnxruntime_pybind11_state.pyd": b"unused Python extension",
        })
        self.assert_manifest(destination, {**libraries, **retained})
        for name in libraries:
            self.assertEqual(list(destination.rglob(name)), [destination / name])
        for original in nested:
            self.assertFalse((destination / original).exists())

    def test_windows_accepts_already_rooted_libraries(self):
        files = {
            "onnxruntime.dll": b"ORT core",
            "onnxruntime_providers_cuda.dll": b"CUDA provider",
            "onnxruntime_providers_shared.dll": b"shared provider",
            "onnxruntime_providers_webgpu.dll": b"WebGPU provider",
        }
        self.assert_manifest(self.prepare(files), files)

    def test_linux_preserves_versioned_libraries(self):
        files = {
            "onnxruntime/capi/libonnxruntime.so.1.30.0": b"ORT core",
            "onnxruntime/capi/libonnxruntime_providers_cuda.so": b"CUDA provider",
            "onnxruntime/capi/libonnxruntime_providers_shared.so": b"shared provider",
            "onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.so": b"WebGPU provider",
            "onnxruntime/LICENSE": b"ORT license",
        }
        destination = self.prepare(files, system="Linux", machine="x86_64")
        self.assert_manifest(destination, {
            **files,
            "libonnxruntime.so": b"ORT core",
            "libonnxruntime_providers_cuda.so": b"CUDA provider",
            "libonnxruntime_providers_shared.so": b"shared provider",
            "libonnxruntime_providers_webgpu.so": b"WebGPU provider",
        })


if __name__ == "__main__":
    unittest.main()
