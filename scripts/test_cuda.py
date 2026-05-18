try:
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(1000, 1000).cuda()
        y = x @ x
        print("CUDA test result:", y.shape)
    else:
        print("CUDA is not available.")

except ImportError:
    print("PyTorch is not installed.")