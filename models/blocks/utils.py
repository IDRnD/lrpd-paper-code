
def get_padding(kernel_size):
    k = kernel_size
    padding = None
    if isinstance(k, tuple) or isinstance(k, list):
        assert ((k[0] % 2) == 1) and ((k[1] % 2) == 1)
        padding = (k[0] // 2, k[1] // 2)
    elif isinstance(k, int):
        assert (k % 2) == 1
        padding = k // 2
    else:
        raise NotImplemented()
    return padding
