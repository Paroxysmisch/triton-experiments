@contextlib.contextmanager
def autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    # Implement the logic for the context manager
    # Use the Triton API to create the context manager
    with triton.autocast(device_type, enabled, dtype, cache_enabled):
        yield
