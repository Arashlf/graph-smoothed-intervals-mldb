import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import ChebConv
from tqdm.auto import tqdm, trange


class TemporalConvolution(nn.Module):
    def __init__(self, inputs, outputs, kernel=3):
        super().__init__()
        self.convolution = nn.Conv2d(inputs, outputs, kernel_size=(kernel, 1))

    def forward(self, values):
        return F.relu(self.convolution(values.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)


class GraphConvolution(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.convolution = ChebConv(inputs, outputs, K=3)

    def forward(self, values, edges, weights):
        batch, steps, nodes, channels = values.shape
        count = batch * steps
        offsets = torch.arange(count, device=values.device) * nodes
        repeated = edges.unsqueeze(0) + offsets.view(count, 1, 1)
        repeated = repeated.permute(1, 0, 2).reshape(2, -1)
        output = self.convolution(values.reshape(count * nodes, channels),
                                  repeated, weights.repeat(count))
        return output.reshape(batch, steps, nodes, -1)


class STGCNBlock(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.temporal_in = TemporalConvolution(inputs, 32)
        self.graph = GraphConvolution(32, 16)
        self.temporal_out = TemporalConvolution(16, 32)

    def forward(self, values, edges, weights):
        values = self.temporal_in(values)
        values = F.relu(self.graph(values, edges, weights))
        return self.temporal_out(values)


class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_one = STGCNBlock(1)
        self.block_two = STGCNBlock(32)
        self.output = nn.Linear(4 * 32, 12)

    def forward(self, values, edges, weights):
        values = self.block_two(self.block_one(values, edges, weights), edges, weights)
        return self.output(values.permute(0, 2, 1, 3).flatten(start_dim=2)).permute(0, 2, 1)


class TrafficWindows(Dataset):
    def __init__(self, data, split):
        self.data = data
        start, end = data["boundaries"][split:split + 2]
        self.starts = np.arange(max(start, 12), end - 12 + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        observed = self.data["observed"][start:start + 12].copy()
        target = self.data["speed"][start:start + 12].copy()
        target[~observed] = 0
        complete = self.data["observed"][start - 12:start].all(axis=0)
        inputs = self.data["inputs"][start - 12:start, :, None].copy()
        return (torch.from_numpy(inputs), torch.from_numpy(target),
                torch.from_numpy(observed), torch.from_numpy(complete))


def train(data, settings, name):
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(settings["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    graph = data["forecast_graph"]
    rows, columns = np.nonzero(graph > 0)
    edges = torch.from_numpy(np.stack((rows, columns))).long().to(device)
    weights = torch.from_numpy(graph[rows, columns]).to(device)
    options = {"batch_size": settings["batch_size"], "num_workers": settings["loader_workers"],
               "pin_memory": device.type == "cuda",
               "persistent_workers": settings["loader_workers"] > 0}
    training = DataLoader(TrafficWindows(data, 0), shuffle=True,
                          generator=torch.Generator().manual_seed(seed), **options)
    validation = DataLoader(TrafficWindows(data, 1), shuffle=False, **options)
    model = STGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    best_loss, stale, best_epoch = float("inf"), 0, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    print(f"{name}: training STGCN on {device}", flush=True)
    epochs = trange(1, settings["epochs"] + 1, desc=name, unit="epoch", dynamic_ncols=True)
    for epoch in epochs:
        losses = []
        for fitting, loader in ((True, training), (False, validation)):
            model.train(fitting)
            total, count = 0.0, 0
            label = "Train" if fitting else "Validate"
            batches = tqdm(loader, desc=f"{label} {epoch}", unit="batch", leave=False,
                           dynamic_ncols=True, mininterval=0.5)
            with torch.set_grad_enabled(fitting):
                for inputs, targets, observed, complete in batches:
                    inputs, targets = inputs.to(device), targets.to(device)
                    mask = observed if fitting else observed & complete[:, None, :]
                    mask = mask.to(device)
                    targets = (targets - data["mean"]) / data["deviation"]
                    if fitting:
                        optimizer.zero_grad(set_to_none=True)
                    predictions = model(inputs, edges, weights)
                    absolute = (predictions - targets).abs()[mask]
                    error_sum = absolute.sum()
                    loss = error_sum / absolute.numel()
                    if fitting:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    total += float(error_sum.detach().cpu())
                    count += absolute.numel()
                    batches.set_postfix(MAE=f"{total / count * data['deviation']:.3f}")
            losses.append(total / count * data["deviation"])
        if losses[1] < best_loss:
            best_loss, best_epoch, stale = losses[1], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        epochs.set_postfix(train=f"{losses[0]:.3f}", validation=f"{losses[1]:.3f}",
                           best=best_epoch, patience=f"{stale}/{settings['patience']}")
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f"{name}: best epoch {best_epoch}, validation MAE {best_loss:.3f} mph", flush=True)
    return model, edges, weights, device


@torch.inference_mode()
def predict(data, fitted, split, batch_size, horizons):
    model, edges, weights, device = fitted
    windows = TrafficWindows(data, split)
    loader = DataLoader(windows, batch_size=batch_size, shuffle=False)
    indices = [horizon // 5 - 1 for horizon in horizons]
    predictions, observations, masks = [], [], []
    for inputs, targets, observed, complete in tqdm(loader, desc=f"Predict split {split}",
                                                   unit="batch", dynamic_ncols=True):
        output = model(inputs.to(device), edges, weights)[:, indices]
        predictions.append(output.mul(data["deviation"]).add(data["mean"]).cpu().numpy())
        observations.append(targets[:, indices].numpy())
        masks.append(observed[:, indices].numpy() & complete[:, None, :].numpy())
    predictions, observations = np.concatenate(predictions), np.concatenate(observations)
    differences = np.abs(predictions - observations)
    differences[~np.concatenate(masks)] = np.nan
    targets = windows.starts[:, None] + np.array(indices)[None, :]
    return {"predictions": predictions, "observations": observations,
            "differences": differences, "targets": targets, "issues": windows.starts - 1}
