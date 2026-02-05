import warnings
warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime, timedelta
from skimage.metrics import structural_similarity as ssim

LEADTIME_HOURS = int(input("LeadTime Hours (multiple of 6): "))

if LEADTIME_HOURS % 6 != 0:
    raise ValueError("LEADTIME_HOURS must be a multiple of 6.")

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = 'data'

CHANNEL_RANGES = {
    'd2m': (190.11102, 305.9109),
    'lsm': (0.0, 1.0),
    'msl': (90174.375, 107080.56),
    'pm1': (0.0, 1.0992195e-05),
    'pm10': (0.0, 3.922431e-05),
    'pm2p5': (0.0, 1.2829033e-05),
    't2m': (190.11142, 325.4682),
    'u10': (-38.362396, 37.811813),
    'v10': (-42.232315, 42.86583),
    'z': (-1230.0354, 55513.16)
}

def normalize(data: np.ndarray, channel: str) -> np.ndarray:
    min_val, max_val = CHANNEL_RANGES[channel]
    max_abs = max(abs(min_val), abs(max_val))
    return data / max_abs

def denormalize(data: np.ndarray, channel: str) -> np.ndarray:
    min_val, max_val = CHANNEL_RANGES[channel]
    max_abs = max(abs(min_val), abs(max_val))
    return data * max_abs

FEATURES = ['d2m', 'lsm', 'msl', 'pm1', 'pm10', 'pm2p5', 't2m', 'u10', 'v10', 'z']
files = pd.date_range(f'2024-01-01', f'2024-12-30').strftime('%Y-%m-%d').tolist()
TIMESTAMPS = ['00', '06', '12', '18']
TARGETS = ['pm1', 'pm2p5', 'pm10']

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features=(16, 32, 64, 128, 256), bilinear=True):
        super().__init__()
        f1, f2, f3, f4, f5 = features
        self.inc = DoubleConv(in_channels, f1)
        self.down1 = Down(f1, f2)
        self.down2 = Down(f2, f3)
        self.down3 = Down(f3, f4)
        self.down4 = Down(f4, f5)
        self.down5 = Down(f5, f5)
        self.up1 = Up(f5 + f5, f4, bilinear)
        self.up2 = Up(f4 + f4, f3, bilinear)
        self.up3 = Up(f3 + f3, f2, bilinear)
        self.up4 = Up(f2 + f2, f1, bilinear)
        self.up5 = Up(f1 + f1, f1, bilinear)
        self.outc = nn.Conv2d(f1, out_channels, kernel_size=1)
        self.activation = nn.Tanh()
    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.size()
            x = x.view(B, T * C, H, W)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        u1 = self.up1(x6, x5)
        u2 = self.up2(u1, x4)
        u3 = self.up3(u2, x3)
        u4 = self.up4(u3, x2)
        u5 = self.up5(u4, x1)
        out = self.outc(u5)
        out = self.activation(out)
        return out

MODELS = {}
for TARGET in TARGETS:
    MODELS[TARGET] = {}
    for grid in range(10):
        model = UNet2D(in_channels=len(FEATURES), out_channels=1).to(device)
        checkpoint = torch.load(f'models/unet_{LEADTIME_HOURS}_{TARGET}_{grid}.pt', map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        MODELS[TARGET][grid] = model

def calculate_metrics(tgt: np.ndarray, pred: np.ndarray, delta: float = 1.0) -> dict:
    x_tgt, x_pred = tgt.flatten(), pred.flatten()
    tgt_mean, pred_mean = np.mean(x_tgt), np.mean(x_pred)
    mean_diff = pred_mean - tgt_mean
    unbiased_pred = x_pred - mean_diff
    crmse = np.sqrt(np.mean((unbiased_pred - x_tgt) ** 2))
    diff = x_pred - x_tgt
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    global_min, global_max = min(pred.min(), tgt.min()), max(pred.max(), tgt.max())
    if global_max > global_min:
        pred_n = (pred - global_min) / (global_max - global_min)
        tgt_n  = (tgt  - global_min) / (global_max - global_min)
        ssim_val = ssim(pred_n, tgt_n, data_range=1.0)
    else:
        ssim_val = np.nan
    return {'crmse': crmse, 'rmse': rmse, 'ssim': ssim_val}

def combine(mats, overlap=15):
    h, w = mats[0].shape
    rows, cols = 2, 5
    step = h - overlap
    out = np.zeros((rows*h - overlap, cols*w - (cols-1)*overlap), float)
    cnt = np.zeros_like(out)
    for i, m in enumerate(mats):
        y = (i // cols) * step
        x = (i % cols) * step
        out[y:y+h, x:x+w] += m
        cnt[y:y+h, x:x+w] += 1
    return out / cnt

Summary_CSV_PTH = f'metrics/unet_{LEADTIME_HOURS}.csv'
summary_results = []
for ts in TIMESTAMPS:
    # CSV_PTH = f'metrics/unet_{LEADTIME_HOURS}_{ts}.csv'
    results = []
    for file in tqdm(files):
        inp, tgt, pred = [], [], []
        ds = np.load(os.path.join(DATA_DIR, f'{file}_{ts}_x.npz'))
        for TARGET in TARGETS:
            inp.append(ds[TARGET])
        ds = np.load(os.path.join(DATA_DIR, (datetime.strptime(f'{file}_{ts}', '%Y-%m-%d_%H') + timedelta(hours=LEADTIME_HOURS)).strftime('%Y-%m-%d_%H_x.npz')))
        for TARGET in TARGETS:
            tgt.append(ds[TARGET])
        for TARGET in TARGETS:
            res = []
            for grid in range(10):
                model = MODELS[TARGET][grid]
                inp_patch = np.load(os.path.join(DATA_DIR, f'{file}_{ts}_{grid}.npz'))
                inp_patch = np.stack([normalize(inp_patch[f], f) for f in FEATURES], axis=0)
                inp_patch = torch.tensor(inp_patch, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(inp_patch).squeeze(0).squeeze(0).cpu().numpy()
                output = denormalize(output[32:224, 32:224], TARGET)
                res.append(output)
            pred_patch = combine(res, overlap=15) + inp[TARGETS.index(TARGET)]
            pred.append(pred_patch)
        inp, tgt, pred = np.array(inp), np.array(tgt), np.array(pred)
        assert inp.shape == (3, 369, 900)
        assert tgt.shape == (3, 369, 900)
        assert pred.shape == (3, 369, 900)
        for i, TARGET in enumerate(TARGETS):
            record = {'target': TARGET, 'file': file}
            metrics = calculate_metrics(tgt=tgt[i], pred=pred[i])
            record.update(metrics)
            results.append(record)
    df = pd.DataFrame(results)
    # df.to_csv(CSV_PTH, index=False)
    for TARGET in TARGETS:
        df_k = df[df["target"] == TARGET]
        temp_record = {'target': TARGET, 'timestamp': ts}
        metric_cols = [c for c in df.columns if c not in ("file", "target")]
        for c in metric_cols:
            temp_record[c] = df_k[c].mean()
        summary_results.append(temp_record)
df_summary = pd.DataFrame(summary_results)
metric_cols = [c for c in df_summary.columns if c not in ("target", "timestamp")]

for TARGET in TARGETS:
    df_k = df_summary[df_summary["target"] == TARGET]
    temp_record = {'target': TARGET, 'timestamp': 'x'}
    temp_record.update(df_k[metric_cols].mean().to_dict())
    summary_results.append(temp_record)
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(Summary_CSV_PTH, index=False)
