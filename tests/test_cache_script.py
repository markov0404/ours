import subprocess, sys, torch, pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
@pytest.mark.timeout(900)
def test_cache_script_resumable(tmp_path):
    cmd = [sys.executable, "scripts/cache_dataset.py", "--dataset", "jevbench", "--split", "public", "--backbone", "Qwen/Qwen3-0.6B",
           "--style", "numeric", "--dtype", "bf16", "--limit-items", "6", "--chunk", "3", "--cache-dir", str(tmp_path)]
    subprocess.run(cmd, check=True); first = list(tmp_path.glob("*.parquet")); assert len(first) == 1
    out = subprocess.run(cmd, check=True, capture_output=True, text=True); assert "0 ítems pendientes" in out.stdout
