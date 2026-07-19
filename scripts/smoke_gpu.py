import sys

import torch


def main() -> int:
    print(f"python: {sys.version}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda: {torch.version.cuda}")

    if not torch.cuda.is_available():
        return 1

    device = torch.device("cuda")
    print(f"device count: {torch.cuda.device_count()}")
    print(f"device name: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")

    x = torch.randn(2048, 2048, device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"matmul checksum: {float(y.mean().cpu()):.6f}")
    print("gpu smoke test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
