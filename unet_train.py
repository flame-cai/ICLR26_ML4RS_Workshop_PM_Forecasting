import warnings
warnings.filterwarnings("ignore")
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torchmetrics.functional import structural_similarity_index_measure as ssim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.empty_cache()
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = False

GRID = input('Grid [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: ')
TARGET = input('Target [pm1, pm10, pm2p5]: ').lower()
LEADTIME_HOURS = int(input('LeadTime Hours (multiple of 6): '))

if GRID not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    raise ValueError("Invalid GRID selection. Choose from [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].")
if TARGET not in ['pm1', 'pm10', 'pm2p5']:
    raise ValueError("Invalid TARGET selection. Choose from ['pm1', 'pm10', 'pm2p5'].")
if LEADTIME_HOURS % 6 != 0:
    raise ValueError("LEADTIME_HOURS must be a multiple of 6.")

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = 'data'

INPUT_SIZE = 256
OUTPUT_SIZE = 192
SPLIT = 0.1
NUM_WORKERS = 12
TEST_YEAR = 2024

BATCH_SIZE = 16
EPOCHS = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 16

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

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(f"{GRID}.npz") and f.startswith(str(TEST_YEAR))==False)
files = files[:-(LEADTIME_HOURS//6)]
n = len(files)
val = int(n * SPLIT)
train = n - val
train, val = random_split(files, [train, val], generator=torch.Generator().manual_seed(SEED))
print(f'Total files: {n}, Train: {len(train)}, Val: {len(val)}')

class CAMSDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        timestamp = datetime.strptime(self.data[idx][:13], '%Y-%m-%d_%H')
        inputs = np.load(os.path.join(DATA_DIR, self.data[idx]))
        inputs = np.stack([normalize(inputs[feat], feat) for feat in FEATURES], axis=0)
        tagret = np.load(os.path.join(DATA_DIR, (timestamp + timedelta(hours=LEADTIME_HOURS)).strftime('%Y-%m-%d_%H') + self.data[idx][13:]))
        target = normalize(tagret[TARGET], TARGET)[32:224, 32:224] - inputs[FEATURES.index(TARGET), 32:224, 32:224]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

train = CAMSDataset(train)
val = CAMSDataset(val)

train = DataLoader(train, batch_size=BATCH_SIZE, sampler=RandomSampler(train), num_workers=NUM_WORKERS, pin_memory=True)
val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

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
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
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

model = UNet2D(in_channels=len(FEATURES), out_channels=1).to(device)

huber = nn.HuberLoss().to(device)

def batch_minmax(x, y):
    g_min = torch.minimum(x.amin(dim=(1,2), keepdim=True),
                           y.amin(dim=(1,2), keepdim=True))
    g_max = torch.maximum(x.amax(dim=(1,2), keepdim=True),
                           y.amax(dim=(1,2), keepdim=True))
    x_n = (x - g_min) / (g_max - g_min)
    y_n = (y - g_min) / (g_max - g_min)
    return x_n.unsqueeze(1), y_n.unsqueeze(1)

def criterion(pred, target, input):
    target = input[:, FEATURES.index(TARGET), 32:224, 32:224] + target.squeeze(1)
    pred = input[:, FEATURES.index(TARGET), 32:224, 32:224] + pred.squeeze(1)
    logcosh_loss = torch.log(torch.cosh(pred - target)).mean()
    huber_loss = huber(pred, target)
    mae = torch.mean(abs(pred - target))
    pred, target = batch_minmax(pred, target)
    ssim_val = ssim(pred, target)
    ssim_loss = 1 - ssim_val
    return 1024 * logcosh_loss + 1024 * huber_loss + 0.0001 * ssim_loss + 0.01 * mae

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

counter = 0
best_val_loss = float('inf')
bar = tqdm(range(EPOCHS), desc="Training")
for epoch in bar:
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(train):
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        outputs = model(inputs)
        outputs = outputs[:, :, 32:224, 32:224]
        loss = criterion(outputs, targets, inputs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= len(train)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val):
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = model(inputs)
            outputs = outputs[:, :, 32:224, 32:224]
            loss = criterion(outputs, targets, inputs)
            val_loss += loss.item()
    val_loss /= len(val)
    scheduler.step()

    bar.set_postfix(train_loss=f"{train_loss}", val_loss=f"{val_loss}")
    
    if val_loss < best_val_loss:
        counter = 0
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'models/unet_{LEADTIME_HOURS}_{TARGET}_{GRID}.pt')
    else:
        counter += 1
        if counter >= PATIENCE:
            print('Early stopping triggered at epoch', epoch)
            break
print('DONE')
