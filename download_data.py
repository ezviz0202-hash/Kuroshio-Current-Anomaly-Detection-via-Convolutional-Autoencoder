import os
import subprocess
from pathlib import Path

DATASET_ID  = "cmems_mod_glo_phy_my_0.083deg_P1D-m"   
VARIABLES   = ["uo", "vo"]          
DEPTH_MIN   = 0.49402499198913574               
DEPTH_MAX   = 0.49402499198913574
LON_MIN, LON_MAX = 130.0, 145.0
LAT_MIN, LAT_MAX = 25.0,  40.0
YEARS       = range(2010, 2021)
OUTPUT_DIR  = Path("data/raw")

def download_year(year: int) -> None:
    out_path = OUTPUT_DIR / f"kuroshio_uv_{year}.nc"
    if out_path.exists():
        print(f"[skip] {out_path} already exists.")
        return

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id",      DATASET_ID,
        "--variable",        "uo",
        "--variable",        "vo",
        "--minimum-longitude", str(LON_MIN),
        "--maximum-longitude", str(LON_MAX),
        "--minimum-latitude",  str(LAT_MIN),
        "--maximum-latitude",  str(LAT_MAX),
        "--minimum-depth",     str(DEPTH_MIN),
        "--maximum-depth",     str(DEPTH_MAX),
        "--start-datetime",    f"{year}-01-01T00:00:00",
        "--end-datetime",      f"{year}-12-31T23:59:59",
        "--output-filename",   str(out_path),
        "--force-download",
    ]

    print(f"[download] {year} → {out_path}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for year {year}.")
    print(f"[done] {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        download_year(year)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
