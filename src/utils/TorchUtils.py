import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        print(f"Running on your {gpu_name} (GPU)")
    else:
        device = torch.device("cpu")
        print("Running on your CPU")
    return device
