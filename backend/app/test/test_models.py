import torch
from app.models.registry import ModelRegistry

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    registry = ModelRegistry.get_instance()

    # 🔥 This is the line you were asking about
    registry.load_all("models", device_str=str(device))

    # Check status
    status = registry.status()
    print("\nModel status:")
    for name, info in status.items():
        print(f"{name}: {info}")

if __name__ == "__main__":
    main()