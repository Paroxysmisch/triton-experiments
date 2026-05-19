import torch

@contextlib.contextmanager
def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    assert device_type == "cuda", "Only 'cuda' is currently supported for autocast"
    assert enabled, "Autocast is always enabled when used as a context manager"
    assert cache_enabled, "Autocast cache is always enabled when used as a context manager"
    yield
