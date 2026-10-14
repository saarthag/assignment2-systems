import torch
import torch.nn.functional as F
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("fc1", f"{x.dtype=}")
        x = self.ln(x)
        print("ln", f"{x.dtype=}")
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = ToyModel(8, 20).to("cuda")
    x = torch.randn(5, 8, device="cuda")
    y = torch.randn(5, 20, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        print(f"{x.dtype=}")

        param_dtypes = [p.dtype for p in model.parameters()]
        print(f"{param_dtypes=}")

        yy = model(x)
        print(f"{yy.dtype=}")

        loss = F.mse_loss(yy, y)
        print(f"{loss.dtype=}")
        loss.backward()

        grad_dtypes = [p.grad.dtype for p in model.parameters() if p.grad is not None]
        print(f"{grad_dtypes=}")
