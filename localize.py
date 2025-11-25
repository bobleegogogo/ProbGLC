import os
import math
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from plonk.pipe import PlonkPipeline  # <-- your patched pipeline with device fixes

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

# === CONFIG ===
data_folder = "data/MultiDisaster_20250320"
metadata_xlsx = os.path.join(data_folder, "Other/allData_20250320.xlsx")
batch_size = 128
thresholds_km = [1, 25, 50, 200, 750, 2500]
num_gpus = torch.cuda.device_count()
output_dir = "localizability_outputs"
os.makedirs(output_dir, exist_ok=True)

# Datasets
datasets = {
    "20": os.path.join(data_folder, "RSI_SVI_UAV_VGI/20%test_RSI_SVI_UAV_VGI.csv"),
    "30": os.path.join(data_folder, "RSI_SVI_UAV_VGI/30%test_RSI_SVI_UAV_VGI.csv"),
}

# Models
models = {
    "20": [
        "nicolas-dufour/PLONK_OSV_5M",
        "nicolas-dufour/PLONK_YFCC",
        "multidisaster20osm",
        "multidisaster20yfcc",
    ],
    "30": [
        "nicolas-dufour/PLONK_OSV_5M",
        "nicolas-dufour/PLONK_YFCC",
        "multidisaster30osm",
        "multidisaster30yfcc",
    ],
}


# === HELPERS ===
def round_coords(lat, lon):
    return int(round(lat)), int(round(lon))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def process_chunk(args):
    chunk_tasks, chunk_gt_latlons, chunk_gt_cells, chunk_filenames, gpu_id, model_name = args

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    pipeline = PlonkPipeline(model_name, device=device)

    out = []
    for i in range(0, len(chunk_tasks), batch_size):
        paths = chunk_tasks[i : i + batch_size]
        cells = chunk_gt_cells[i : i + batch_size]
        gtll = chunk_gt_latlons[i : i + batch_size]
        fns = chunk_filenames[i : i + batch_size]

        imgs = []
        for p in paths:
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)

        preds = pipeline(imgs, batch_size=len(imgs))

        for img, fname, gt_cell, gt_latlon, (pred_lat, pred_lon) in zip(
            imgs, fns, cells, gtll, preds
        ):
            dist_km = haversine(gt_latlon[0], gt_latlon[1], pred_lat, pred_lon)
            try:
                loc = pipeline.compute_localizability(img).item()
            except Exception as e:
                print(f"[GPU {gpu_id}] localizability failed for {fname}: {e}")
                loc = float("nan")

            row = {
                "filename": fname,
                "gt_lat": gt_latlon[0],
                "gt_lon": gt_latlon[1],
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "gt_cell_lat": gt_cell[0],
                "gt_cell_lon": gt_cell[1],
                "distance_km": dist_km,
                "localizability": loc,
            }
            for thr in thresholds_km:
                row[f"within_{thr}km"] = dist_km <= thr
            out.append(row)
    return out


# === MAIN RUNNER ===
def run_split(split_name, test_csv, model_list):
    print(f"\n=== Processing split {split_name} ===")

    test_df = pd.read_csv(test_csv, header=None)
    metadata_df = pd.read_excel(metadata_xlsx)
    vgi_column_index = 3
    metadata_df["filename"] = metadata_df["newnewID"].astype(str) + ".jpg"

    tasks, gt_cells, gt_latlons, filenames = [], [], [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        vgi_file = row[vgi_column_index]
        vgi_id = int(vgi_file.split(".")[0].replace("VGI", "").replace("\\", ""))
        gt_row = metadata_df[metadata_df["newnewID"] == vgi_id]
        if gt_row.empty:
            continue
        gt_lat = gt_row["Latitude"].values[0]
        gt_lon = gt_row["Longitude"].values[0]
        gt_cell = round_coords(gt_lat, gt_lon)
        img_path = os.path.join(data_folder, vgi_file.replace("\\", "/"))
        if not os.path.isfile(img_path):
            continue
        tasks.append(img_path)
        gt_cells.append(gt_cell)
        gt_latlons.append((gt_lat, gt_lon))
        filenames.append(vgi_file)

    for model_name in model_list:
        print(f"\n--- Running {model_name} on split {split_name} ---")
        chunks = [[] for _ in range(num_gpus)]
        for idx, (t, gtc, gtl, fn) in enumerate(zip(tasks, gt_cells, gt_latlons, filenames)):
            chunks[idx % num_gpus].append((t, gtl, gtc, fn))

        args_list = []
        for gpu_id, chunk in enumerate(chunks):
            if not chunk:
                continue
            t, gtl, gtc, fn = zip(*chunk)
            args_list.append((list(t), list(gtl), list(gtc), list(fn), gpu_id, model_name))

        with Pool(processes=len(args_list)) as pool:
            all_results = pool.map(process_chunk, args_list)

        df = pd.DataFrame([r for part in all_results for r in part])
        if len(df) == 0:
            print(f"No results for {model_name} on {split_name}")
            continue

        out_file = os.path.join(output_dir, f"{model_name.replace('/','_')}_{split_name}.csv")
        df.to_csv(out_file, index=False)
        print(f"✅ Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    for split_name, test_csv in datasets.items():
        run_split(split_name, test_csv, models[split_name])

