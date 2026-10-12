import importlib
import pytest

def test_imports():
    for m in ["numpy", "scipy", "sklearn", "pyarrow", "datasets", "faiss", "torch", "transformers", "sentence_transformers", "fastapi"]:
        importlib.import_module(m)

def test_torch_cuda_version():
    import torch
    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    assert (major, minor) >= (2, 6), torch.__version__
    if not torch.cuda.is_available():
        pytest.skip("sin GPU: dec funciona en CPU (device='cpu'); los tests de GPU se saltan")
    assert torch.cuda.device_count() >= 1
