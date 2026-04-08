import glob
import os
from pathlib import Path

import numpy as np
import xarray as xr

RAW_DIR   = Path("data/raw")
OUT_DIR   = Path("data/processed")

# Training on quiescent years 2010–2016, val 2017–2018, test 2019–2020
TRAIN_YEARS = set(range(2010, 2017))
VAL_YEARS   = {2017, 2018}
TEST_YEARS  = {2019, 2020}

U_VAR = "uo"   # GLORYS12v1 eastward velocity variable name
V_VAR = "vo"   # northward velocity variable name

def load_yearly_files() -> xr.Dataset:
    files = sorted(RAW_DIR.glob("kuroshio_uv_*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {RAW_DIR}. "
                                "Run download_data.py first.")
    print(f"Loading {len(files)} file(s)...")
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    return ds


def build_land_mask(u_arr: np.ndarray) -> np.ndarray:
   
    mask = ~np.all(np.isnan(u_arr), axis=0)   
    return mask


def normalise(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / (std + 1e-8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_yearly_files()
    times = ds.time.values  

    u_raw = ds[U_VAR].values.squeeze()   
    v_raw = ds[V_VAR].values.squeeze()

    if u_raw.ndim == 4:       
        u_raw = u_raw[:, 0]
        v_raw = v_raw[:, 0]

    N, H, W = u_raw.shape
    print(f"Dataset shape: {N} frames × {H} × {W} grid")

    land_mask = build_land_mask(u_raw)          # (H, W), True = ocean
    print(f"Ocean pixels: {land_mask.sum()} / {H * W}")

    years = np.array([int(str(t)[:4]) for t in times])
    train_idx = np.where(np.isin(years, list(TRAIN_YEARS)))[0]
    val_idx   = np.where(np.isin(years, list(VAL_YEARS)))[0]
    test_idx  = np.where(np.isin(years, list(TEST_YEARS)))[0]
    print(f"Split sizes — train: {len(train_idx)}, "
          f"val: {len(val_idx)}, test: {len(test_idx)}")

    u_train = u_raw[train_idx]
    v_train = v_raw[train_idx]

    ocean_u = u_train[:, land_mask]
    ocean_v = v_train[:, land_mask]

    u_mean, u_std = float(np.nanmean(ocean_u)), float(np.nanstd(ocean_u))
    v_mean, v_std = float(np.nanmean(ocean_v)), float(np.nanstd(ocean_v))
    print(f"u: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"v: mean={v_mean:.4f}, std={v_std:.4f}")

    u_norm = normalise(u_raw, u_mean, u_std)
    v_norm = normalise(v_raw, v_mean, v_std)
    u_norm = np.nan_to_num(u_norm, nan=0.0)
    v_norm = np.nan_to_num(v_norm, nan=0.0)

    frames = np.stack([u_norm, v_norm], axis=1).astype(np.float32)
    print(f"Final tensor shape: {frames.shape}")

    np.save(OUT_DIR / "kuroshio_frames.npy", frames)
    np.save(OUT_DIR / "land_mask.npy", land_mask)
    np.savez(OUT_DIR / "norm_stats.npz",
             u_mean=u_mean, u_std=u_std,
             v_mean=v_mean, v_std=v_std)
    np.savez(OUT_DIR / "split_indices.npz",
             train=train_idx, val=val_idx, test=test_idx)
    print(f"\nPreprocessing complete. Files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
