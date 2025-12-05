import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT = Path("data")
PROC_DIR = ROOT / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TEST_FOLDERS = [
    ROOT / "1st_test" / "1st_test",
    ROOT / "2nd_test" / "2nd_test",
    ROOT / "3rd_test" / "3rd_test",
]

def load_and_segment(file, window_size=512):
    """Read a vibration signal file and split it into fixed-length segments."""
    try:
        df = pd.read_csv(file, header=None, sep=r"\s+")
    except Exception:
        df = pd.read_csv(file, header=None)
    sig = df.iloc[:, 0].values.astype(np.float32)
    n = len(sig) // window_size
    sig = sig[:n * window_size]
    return sig.reshape(n, window_size)

def preprocess_all():
    all_segments = []
    for folder in TEST_FOLDERS:
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue
        print(f"Loading data from {folder} ...")
        for file in folder.iterdir():
            if file.is_file():
                try:
                    segs = load_and_segment(file)
                    all_segments.append(segs)
                except Exception as e:
                    print(f"Skipping {file.name}: {e}")

    if not all_segments:
        print("No data found. Make sure your dataset is in the correct folders.")
        return

    X = np.vstack(all_segments)
    print(f"Loaded {X.shape[0]} segments total ({X.shape[1]} samples each).")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=True)

    np.save(PROC_DIR / "bearing_train.npy", X_train)
    np.save(PROC_DIR / "bearing_test.npy", X_test)
    np.save(PROC_DIR / "bearing_all.npy", X)

    print(f"Saved: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
    print(f"Files written to: {PROC_DIR}")

if __name__ == "__main__":
    preprocess_all()