# -*- coding: utf-8 -*-
# =======================================
import os, random
from pathlib import Path
import re
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA

try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except Exception:
    _HAS_RUPTURES = False
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "ruptures"])

# === PATCH-RN: robust per-file normalization (median/MAD) with cache ===
FILE_STATS_CACHE = {}  # key: (fp, stat_hz) -> (mu[C,1], mad[C,1])

# ========= Global Plot Save Dir =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FILE_STATS_CACHE = {}  # (fp, stat_hz, ch_key) -> (mu, mad)

def _ch_key_from_slice(slc):
    if isinstance(slc, slice):
        return ('slice', slc.start, slc.stop, slc.step)
    return tuple(map(int, slc))  

def get_file_stats(fp, stat_hz=10, eps=1e-4):
    # ch_key = (CHANNEL_SLICE.start, CHANNEL_SLICE.stop, CHANNEL_SLICE.step)
    ch_key = _ch_key_from_slice(CHANNEL_SLICE)
    key = (fp, int(stat_hz), ch_key)
    # key = (fp, int(stat_hz))
    cached = FILE_STATS_CACHE.get(key, None)
    if cached is not None:
        return cached

    with h5py.File(fp, "r") as f:
        sig = f["/sig"]            # HDF5 dataset, shape [C, N]
        # Fs  = float(sig.attrs["Fs"])
        Fs = safe_item(f["/sig"].attrs["Fs"]) 
        step = max(1, int(round(Fs / float(stat_hz))))   # ~stat_hz Hz
        # Read only a decimated signal slice.
        # sig_dec = sig[:, ::step]
        sig_dec = sig[CHANNEL_SLICE, ::step]
        mu  = np.median(sig_dec, axis=1, keepdims=True).astype(np.float32)
        mad = np.median(np.abs(sig_dec - mu), axis=1, keepdims=True).astype(np.float32)
        mad = np.maximum(mad, eps)  # Prevent extremely small MAD values from amplifying noise.
    FILE_STATS_CACHE[key] = (mu, mad)
    return mu, mad

# Configure fonts to avoid garbled plot text.
def _setup_chinese_font():
    # Candidate font list in priority order.
    try_fonts = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]

    # Collect available system fonts.
    available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

    # Use the first available preferred font.
    for f in try_fonts:
        if f in available_fonts:
            plt.rcParams['font.family'] = f
            print(f"[INFO] 使用字体: {f}")  # Report the selected font at runtime.
            break
    else:
        plt.rcParams['font.family'] = "sans-serif"
        print("[WARNING] 未找到候选中文字体，使用默认字体，中文可能无法正常显示。")

    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()

# ========= Paths =========
DATA_DIR = r"path\to\filtered_h5_eeg_files"
N_CLASSES = 3
CPD_POS_CLASS = 1
AUC_POS_CLASS = 1
# KEEP_CH_IDX = slice(0, 8)
CHANNEL_SLICE = slice(0, 8)

SELECTED_SUBJECTS = {
    "mir过表达组": ["31号", "33号", "35号", "37号"],   # Subject 31 is missing the 30 min pre-stage-IV recording.
    "pilo组":      ["1号", "2号", "5号", "10号"],
    "sponges组":   ["45号", "46号", "47号"],
    "VPA组":       ["25号"],    # Subject 23 is missing the 30 min pre-stage-IV recording.
}

# Alternative VPA subject list; subject 23 is missing the 30 min pre-stage-IV recording.

def _norm_part(s: str) -> str:
    return str(s).strip().lower()

def _norm(s: str) -> str:
     """小写化并去掉首尾空格，方便匹配"""
     return str(s).strip().lower()

def _contains_token(parts_lower, token: str) -> bool:
    t = _norm(token)
    return any(t in p for p in parts_lower)


def list_h5_by_group_and_subject_root(data_dir: str, selection: dict):
    root = Path(data_dir)
    subj_files_map = {}
    # First pass: collect candidate directories and reduce repeated rglob matching.
    dir_candidates = [p for p in root.rglob("*") if p.is_dir()]
    # Find the most likely subject root directory for each group and subject.
    for grp, subjects in selection.items():
        grp_l = _norm(grp)
        for subj in subjects:
            subj_l = _norm(subj)
            # A candidate subject root must contain both the group name and subject token.
            candidates = []
            for d in dir_candidates:
                parts = [_norm(x) for x in d.parts]
                if any(grp_l in p for p in parts) and any(subj_l in p for p in parts):
                    candidates.append(d)
            if not candidates:
                continue
            # Prefer the shortest directory closest to the group/subject root.
            # Use the number of matched tokens first, then path depth as a secondary score.
            def score_dir(d: Path):
                parts = [_norm(x) for x in d.parts]
                hit = 0
                hit += sum(1 for p in parts if grp_l in p)
                hit += sum(1 for p in parts if subj_l in p)
                return (hit, len(parts))  # Lower hit count and shorter path are preferred.
            candidates = sorted(candidates, key=score_dir)
            subj_root = candidates[0]  # Treat this directory as the subject root.

            # Collect all H5 files below this subject root.
            files = sorted([str(p) for p in subj_root.rglob("*.h5")])
            if files:
                subj_files_map[(grp, subj)] = files

    # Merge all selected H5 files for downstream training and CPD.
    all_files = []
    for _, flist in subj_files_map.items():
        all_files.extend(flist)
    all_files = sorted(set(all_files))

    # Print a selection summary for verification.
    print(f"[SelectStrict] 命中 {len(all_files)} 个 .h5 文件（严格限定为：白名单组/个体的“个体文件夹”范围内）")
    for (grp, subj), flist in sorted(subj_files_map.items()):
        print(f"  - {grp} / {subj}: {len(flist)} files")

    return all_files, subj_files_map

# ========= Windowing =========
WIN_SEC    = 15     # Window length.
STRIDE_SEC = 1.0    # Window stride; smaller values improve localization but increase computation.
DS_STEP    = 1       # No downsampling.

# ========= Sampling / Training =========
MAX_WINDOWS_PER_FILE = 120
TOTAL_FILES_TARGET   = 120
MIN_PER_CLASS        = 3

EPOCHS      = 24  
BATCH_SIZE  = 128
VAL_SPLIT   = 0.2
MAX_TRAIN_BATCHES = 80
NUM_WORKERS = 0

# ========= CPD Params =========
SMOOTH_SEC     = 4.0   # Stronger smoothing; originally 6.0.
TH_HIGH_FIXED  = 0.75   # Increase the hysteresis threshold.
TH_LOW_FIXED   = 0.5
MIN_STATE_SEC  = 4.0   # Minimum state duration; originally 8.0.
MIN_GAP_SEC    = 6.0    # Minimum change-point interval; originally 16.0.

# Parameters for the embedding-distance CPD path.
DIST_SMOOTH_SEC = 1.0       # Smoothing duration for the distance sequence.
DIST_Q          = 0.98      # Quantile threshold for absolute differences.
DIST_BASE_SEC = 5.0
DIST_MIN_STATE  = 1.0
DIST_MIN_GAP    = 0.5

# ========= Plot Params =========
ENABLE_EMBED_CPD = False
HEATMAP_MAX_WIN = 2000
HEATMAP_DPI = 120
PLOT_DS = 4

# ===== Ruptures (feature-space CPD) params =====
USE_RUPTURES = True
RUPTURES_MODEL = "rbf"        # RBF is more robust for high-dimensional embeddings; use L2 for mean shifts.
RUPTURES_BACKEND = "pelt"     # Available ruptures backends.
RUPTURES_PEN = 4.0            # PELT penalty; larger values produce fewer change points.
RUPTURES_N_BKPS = 12          # Maximum number of breakpoints.
RUPTURES_JUMP = 1             # No downsampling.
RUPTURES_MIN_STATE_SEC = MIN_STATE_SEC  # Minimum duration for each segment.

# ========= Reproducibility & Device =========
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def safe_item(x):
    try:
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float(x)

# ==== use labels.csv to override file-level labels ====
CSV_PATH = r"path\to\labels.csv"  # Read filename-level annotations.
LABEL_MAP = None

def _norm_name(s: str) -> str:
    # Lowercase, trim whitespace, and keep only the basename.
    base = os.path.basename(s).strip().lower()
    return base

def _load_labels_from_csv(csv_path):   # Load labels from the labels CSV file.
    import csv, os
    m = {}
    if not os.path.exists(csv_path):
        print(f"[CSV] Not found: {csv_path}")
        return m

    # Remove UTF-8 BOM if present.
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        raw = reader.fieldnames or []
        norm = [(c or "").strip().lower().lstrip("\ufeff") for c in raw]

        def pick(cands):
            for token in cands:
                if token in norm:
                    return raw[norm.index(token)]
            return None

        k_name  = pick(["filename", "file", "name"])
        k_label = pick(["label", "y"])
        if not k_name or not k_label:
            print(f"[CSV] Columns not found. Found={raw}")
            return m

        BLANKS = {"", " ", "na", "n/a", "nan", "null", "-", "--"}  # Treat these values as unlabeled.
        for row in reader:
            fn = (row.get(k_name, "") or "").strip()
            lb = (row.get(k_label, "") or "").strip().lower()
            if not fn or lb in BLANKS:
                # Skip unlabeled rows instead of writing them to the dictionary.
                continue
            try:
                y = int(lb)
            except:
                continue

            base = os.path.basename(fn).strip().lower()
            key  = base
            key2 = os.path.splitext(base)[0]
            m[key]  = y
            m.setdefault(key2, y)

    print(f"[CSV] Loaded {len(m)} labeled entries from {csv_path}")
    return m

def read_label_h5only(fp):
    with h5py.File(fp, "r") as f:
        if "/label" in f:
            return int(np.array(f["/label"])[()])
        return int(np.array(f["/sig"].attrs.get("label", 0)))

CSV_STRICT  = True
CSV_DEFAULT = -1

def read_label(fp):
    """优先用 CSV 覆盖；找不到时回退 H5。"""
    global LABEL_MAP
    if LABEL_MAP is None:
        LABEL_MAP = _load_labels_from_csv(CSV_PATH)
    key = _norm_name(fp)
    key_noext = os.path.splitext(key)[0]
    if key in LABEL_MAP or key_noext in LABEL_MAP:
        return int(LABEL_MAP.get(key, LABEL_MAP.get(key_noext)))

        # Missing CSV entry means unlabeled.
    if CSV_STRICT:
        return CSV_DEFAULT
    else:
        return read_label_h5only(fp)  

def _safe_name(s: str) -> str:
    """把中文/空格/特殊符号替换成安全文件名字符。"""
    s = str(s).strip()
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)  # Characters not allowed by Windows filenames.
    s = s.replace(" ", "_")
    return s

def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        print("[Warn] Only one class in y_true; AUC undefined.")
        return float('nan')
    return roc_auc_score(y_true, y_score)

def list_h5_in_target_subdir(root_dir, subdir_name="空载组"):
    root = Path(root_dir)
    files = []
    for p in root.rglob("*.h5"):
        if subdir_name in [part for part in p.parts]:
            files.append(str(p))
    return sorted(files)

def _build_subject_state_map_from_csv(csv_path):
    import csv
    m = {}
    if not os.path.exists(csv_path):
        return m
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols_raw = reader.fieldnames or []
        cols = [c.strip().lower() for c in cols_raw]
        def pick(*cands):
            for x in cands:
                if x in cols:
                    i = cols.index(x); return cols_raw[i]
            return None
        k_file  = pick("filename","file","path","name")
        k_subj  = pick("subject","rat","id")
        k_state = pick("state","label","class")
        k_order = pick("order_hint","minutes","order","minute")
        for row in reader:
            fn = (row.get(k_file) or "").strip()
            if not fn:
                continue
            base = os.path.basename(fn).lower()
            entry = {
                "subject": (row.get(k_subj)  or "").strip(),
                "state":   (row.get(k_state) or "").strip(),
                "order_hint": row.get(k_order)
            }
            m[base] = entry
            m[os.path.splitext(base)[0]] = entry  # Allow matching without filename extension.
    return m

def _infer_subject_from_name(fp):
    s_full = str(fp)
    for part in Path(s_full).parts[::-1]:
        m = re.search(r'(\d+)\s*号', part.lower())
        if m:
            return f"{m.group(1)}号"
    s = os.path.basename(s_full).lower()
    m = re.search(r'(rat|subject|mouse|m)\s*[_-]?\s*(\d+)', s)
    if m:
        return f"{m.group(2)}号"
    return ""


def parse_state_4buckets(text_or_name: str) -> str:
    """把自由文本映射到四个状态桶名（与你脚本的约定一致）。"""
    s = str(text_or_name).lower()
    if ("建模前" in s) or ("baseline" in s):
        return "状态1_建模前"
    if ("iv级前" in s and "min" in s):
        return "状态2_IV级前"
    if ("止惊前" in s) or ("止驚前" in s) or ("止惊后" in s) or ("止驚後" in s):
        return "状态3_止惊前后"
    if ("建模后" in s) or ("建模後" in s) or re.search(r'\b(1d|3d|7d|28d)\b', s):
        return "状态4_建模后"
    return ""

def _norm_lower(s: str) -> str:
    return os.path.basename(s).lower().replace("－","-").replace("—","-").replace("–","-")

def _minutes_from_name(name: str) -> float:
    # Supports 10min, 20min, 30min, 1h, 2h, and 3h patterns.
    s = name.lower()
    m = re.search(r'(\d+)\s*min', s)
    if m: return float(m.group(1))
    m = re.search(r'(\d+)\s*h', s)
    if m: return float(m.group(1)) * 60.0
    return float("inf")

def group_files_by_subject_then_state(files: list, labels_csv: str = CSV_PATH) -> dict:
    csv_map = _build_subject_state_map_from_csv(labels_csv)
    tmp = []
    for fp in files:
        base = os.path.basename(fp).lower()
        subj = ""
        state_raw = ""
        order_hint = None
        if base in csv_map:
            subj = csv_map[base].get("subject","") or _infer_subject_from_name(fp)
            state_raw = csv_map[base].get("state","") or base
            oh = csv_map[base].get("order_hint")
            try:
                order_hint = float(oh) if oh not in (None,"") else None
            except:
                order_hint = None
        else:
            subj = _infer_subject_from_name(fp)
            state_raw = base
        bucket = parse_state_4buckets(state_raw)
        if not bucket:  # Skip files whose state cannot be parsed.
            continue
        tmp.append((subj, bucket, order_hint, fp))

    # Map each subject to four state buckets.
    out = {}
    for subj, bucket, oh, fp in tmp:
        out.setdefault(subj, {"状态1_建模前":[], "状态2_IV级前":[],
                              "状态3_止惊前后":[], "状态4_建模后":[]})
        out[subj][bucket].append((fp, oh))

    # Sort by CSV order_hint first, then fallback to filename time rules.
    def _minutes_from_name_for_sort(fp):
        return _minutes_from_name(_norm_lower(fp))  # 30>20>10

    def _sort_list(bucket, lst):
        if bucket == "状态2_IV级前":
            # For pre-stage-IV files, sort 30, 20, 10 minutes in descending order.
            return sorted(lst, key=lambda it: (
                float('-inf') if it[1] is None else -float(it[1]),
                -_minutes_from_name_for_sort(it[0])
            ))
        elif bucket == "状态3_止惊前后":
            order = ["止惊前10min","止驚前10min","止惊后10min","止驚後10min",
                     "止惊后1h","止驚後1h","止惊后2h","止驚後2h","止惊后3h","止驚後3h"]
            def key(fp):
                n=_norm_lower(fp)
                for i,t in enumerate(order):
                    if t in n: return i
                return len(order)+1
            return sorted(lst, key=lambda it: (
                float('-inf') if it[1] is None else float(it[1]),  # For this bucket, smaller order_hint values come first.
                key(it[0])
            ))
        elif bucket == "状态4_建模后":
            order = ["建模后1d","建模後1d","建模后3d","建模後3d",
                     "建模后7d","建模後7d","建模后28","建模後28"]
            def key(fp):
                n=_norm_lower(fp)
                for i,t in enumerate(order):
                    if t in n: return i
                return len(order)+1
            return sorted(lst, key=lambda it: (
                float('-inf') if it[1] is None else float(it[1]),
                key(it[0])
            ))
        else:
            # For pre-modeling files, use natural order.
            return sorted(lst, key=lambda it: (
                float('-inf') if it[1] is None else float(it[1]),
                _norm_lower(it[0])
            ))

    for subj in list(out.keys()):
        for b in list(out[subj].keys()):
            out[subj][b] = [fp for (fp, _) in _sort_list(b, out[subj][b])]
    return out

# Load, normalize, and concatenate files; Fs and channel count must match.

def _load_norm_file(fp: str, ds_step: int = 1) -> tuple:
    """
    读取单文件：返回 (sig_norm[C,T], Fs)
    使用脚本里的 get_file_stats(fp, stat_hz=10) 做每文件的 robust 归一化。
    """
    with h5py.File(fp, "r") as f:
        # sig = np.array(f["/sig"])   # (C, N)
        sig = np.array(f["/sig"][CHANNEL_SLICE, :]) 
        Fs  = safe_item(f["/sig"].attrs["Fs"])
        # Fs  = float(f["/sig"].attrs["Fs"])
    if ds_step > 1:
        sig = sig[:, ::ds_step]
    mu, mad = get_file_stats(fp, stat_hz=10)
    # mu, mad = get_file_stats(fp, stat_hz=10)
    # mu, mad = mu[KEEP_CH_IDX, :], mad[KEEP_CH_IDX, :]
    x = np.clip((sig - mu) / mad, -5, 5).astype(np.float32)
    return x, float(Fs)

def concat_files_in_order(file_list: list, ds_step: int = 1) -> tuple:
    xs, names, Ls = [], [], []
    Fs_common = None
    C_common  = None
    for fp in file_list:
        x, Fs = _load_norm_file(fp, ds_step=ds_step)
        if Fs_common is None: Fs_common = Fs
        elif abs(Fs - Fs_common) > 1e-6:
            raise ValueError(f"Fs 不一致：{Fs_common} vs {Fs} ({os.path.basename(fp)})")
        if C_common is None: C_common = x.shape[0]
        elif x.shape[0] != C_common:
            raise ValueError(f"通道数不一致：{C_common} vs {x.shape[0]} ({os.path.basename(fp)})")
        xs.append(x); names.append(os.path.basename(fp)); Ls.append(x.shape[1])

    if not xs:
        return None, None, []
    X = np.concatenate(xs, axis=1) if len(xs) > 1 else xs[0]

    # Compute each file span in the concatenated timeline in seconds.
    t = 0.0; bounds = []
    for nm, L in zip(names, Ls):
        a = t; b = t + L/float(Fs_common)
        bounds.append((nm, a, b))
        t = b
    return X, float(Fs_common), bounds



def check_concat_validity(file_list, sig_cat, Fs, bounds):
    ok = True; notes=[]
    total_sec = sum((b-a) for _,a,b in bounds)
    if sig_cat is None or total_sec <= 0:
        ok=False; notes.append("empty_concat")
    return ok, {"n_files":len(file_list), "dur_sec":total_sec, "Fs":Fs, "notes":";".join(notes)}

def _draw_cp_vlines(ax, cps, label="CP(final)", color="red", lw=1.6, alpha=0.95):
    if not cps: 
        return
    # Add the legend label only once to avoid duplicates.
    first = True
    xmin, xmax = ax.get_xlim()
    for t in sorted(float(x) for x in cps):
        if t < xmin or t > xmax:
            continue  # Skip change points outside the current axis range.
        ax.axvline(t, color=color, lw=lw, alpha=alpha,
                   zorder=9, label=(label if first else None))
        first = False

import math

def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size - (m - 1) * tau
    if n <= 0:
        return np.nan
    
    Y = np.vstack([x[i:i + n] for i in range(0, m * tau, tau)]).T  # [n, m]
    orders = np.argsort(Y, axis=1, kind="mergesort")
    # patterns = np.apply_along_axis(lambda r: tuple(np.argsort(r)), 1, Y)
    
    from collections import Counter
    cnt = Counter(map(tuple, orders))
    p = np.array(list(cnt.values()), dtype=np.float64)
    p /= p.sum() + eps
    H = -np.sum(p * np.log(p + eps))
    # Hmax = math.log(np.math.factorial(m))
    Hmax = math.log(math.factorial(int(m)))
    return float(H / (Hmax + eps))


def plot_prob_with_cps(times, p_smooth, th_enter, th_exit, cps_final,
                       bounds=None, title="", save_path=None, cps_raw=None, spans=None):
    idx = slice(None, None, max(1, int(PLOT_DS)))
    t_plot = np.asarray(times)[idx]
    p_plot = np.asarray(p_smooth)[idx] 

    fig, ax = plt.subplots(figsize=(12,5))

    if spans:
        firstA, firstB = True, True
        for name, a, b, color, alpha in spans:
            ax.axvspan(a, b, color=color, alpha=alpha, zorder=0,
                       label=name if ((name.endswith("A") and firstA) or (name.endswith("B") and firstB) or (name not in ("Block A","Block B"))) else None)
            if name.endswith("A"): firstA = False
            if name.endswith("B"): firstB = False
    ax.plot(times, p_smooth, label="p(smooth)")
    # ax.axhline(th_enter, linestyle="--", label=f"TH_ENTER={th_enter:.2f}")
    # ax.axhline(th_exit,  linestyle=":",  label=f"TH_EXIT={th_exit:.2f}")

    if bounds:
        for _, a, b in bounds:
            ax.axvspan(a, b, color="gray", alpha=0.07, zorder=0)

    if cps_raw:
        first = True
        for t in sorted(float(x) for x in cps_raw):
            ax.axvline(t, color="orange", lw=1.2, ls="--", alpha=0.9,
                       zorder=9, label=("CP(candidates)" if first else None))
            first = False

    # Final change points.
    _draw_cp_vlines(ax, cps_final, label="CP(final)", color="red", lw=1.6, alpha=0.95)

    ax.set_xlabel("Time (s)"); ax.set_ylabel(f"Prob(class={CPD_POS_CLASS})")
    ax.set_title(title or "Changepoint detection (t-test filtered)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    
    if save_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(PLOT_DIR, "plot_prob.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=HEATMAP_DPI)
    plt.close(fig)
    print(f"[Saved] {save_path}")

# def plot_signal_with_cps(sig_cat, Fs, cps_final,
#                          ch_idx=0, bounds=None, title="", save_path=None, spans=None):
# Concatenated normalized signal prepared by _run_cpd_on_concat.
# Sampling rate and final change-point times in seconds.
#     C, T = sig_cat.shape
#     t = np.arange(T, dtype=np.float64) / float(Fs)

# Lightly downsample for plotting to avoid overly dense figures.
#     step = max(1, int(round(Fs / 250.0)))
#     y = sig_cat[int(ch_idx), ::step]
#     tx = t[::step]

#     fig, ax = plt.subplots(figsize=(12, 5))

#     if spans:
#         firstA, firstB = True, True
#         for name, a, b, color, alpha in spans:
#             ax.axvspan(a, b, color=color, alpha=alpha, zorder=0,
#                        label=name if ((name.endswith("A") and firstA) or (name.endswith("B") and firstB) or (name not in ("Block A","Block B"))) else None)
#             if name.endswith("A"): firstA = False
#             if name.endswith("B"): firstB = False

#     if bounds:
#         for _, a, b in bounds:
#             ax.axvspan(a, b, color="gray", alpha=0.07, zorder=0)

#     ax.plot(tx, y, lw=0.8, label=f"ch{ch_idx} signal")

#     _draw_cp_vlines(ax, cps_final, label="CP(final)", color="red", lw=1.6, alpha=0.95)

#     ax.set_xlabel("Time (s)")
# Set the y-axis label to signal amplitude in uV.
#     ax.set_title(title or "Changepoint detection on raw signal")
#     ax.legend(loc="lower right")
#     fig.tight_layout()

#     if save_path is None:
#         os.makedirs(PLOT_DIR, exist_ok=True)
#         save_path = os.path.join(PLOT_DIR, "plot_signal.png")
#     else:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     fig.savefig(save_path, dpi=HEATMAP_DPI)
#     plt.close(fig)
#     print(f"[Saved] {save_path}")


def plot_channel_energy_heatmap(sig_cat, Fs, cps, win_sec=2.0, hop_sec=0.5, title="", save_path=None, cmap="seismic",spans=None):
    C,T = sig_cat.shape
    win = int(win_sec*Fs); hop=int(hop_sec*Fs)
    if win <= 0 or T < win:
        print("[Heatmap] 序列太短，跳过绘图。")
        return
    starts = list(range(0, max(1,T-win+1), max(1,hop)))
    Nwin = len(starts)
    if Nwin == 0:
        print("[Heatmap] 时间轴为空，跳过绘图。")
        return
    HEATMAP_MAX_WIN = 4000  # Reduce this value if needed.
    if Nwin > HEATMAP_MAX_WIN:
        step = int(np.ceil(Nwin / HEATMAP_MAX_WIN))
        starts = starts[::step]
        Nwin = len(starts)
    mats=[]; times=[]
    for s in starts:
        seg = sig_cat[:, s:s + win]
        bp = np.mean(seg.astype(np.float32) ** 2, axis=1, dtype=np.float32)  # Broadband energy.
        mats.append(bp)
        times.append((s + win / 2) / Fs)
    if len(times) == 0:
        print("[Heatmap] 时间轴为空，跳过绘图。")
        return
    M = np.vstack(mats).T.astype(np.float32)  # [C, Nwin]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    if spans:
        for name, a, b, color, alpha in spans:
            ax.axvspan(a, b, color=color, alpha=alpha, zorder=0)
    im = ax.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, C],
        cmap=cmap,
        rasterized=True,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Per-channel energy")

    _draw_cp_vlines(ax, cps)  # Overlay red change-point lines.
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title or "Channel-wise energy (+CP)")
    fig.tight_layout()
    if save_path is None:
        os.makedirs(PLOT_DIR, exist_ok=True)
        save_path = os.path.join(PLOT_DIR, "heatmap.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def _read_1s_segment(fp: str, ch_idx: int, sec: float = 1.0, where: str = "middle"):
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])  # [C,N]
        Fs = safe_item(f["/sig"].attrs["Fs"])
    C, N = sig.shape
    ch = int(ch_idx)
    L = int(round(sec * Fs))
    if L <= 0 or L > N:
        L = min(N, int(Fs))
    if where == "start":
        s = 0
    elif where == "end":
        s = max(0, N - L)
    else:
        s = max(0, (N - L) // 2)
    seg = sig[ch, s:s + L]
    t = np.arange(seg.size) / Fs * 1000.0  # ms
    return t, seg, Fs

def plot_subject_three_phases(subject: str, ch_idx: int = 0, sec: float = 1.0,
                              prefer_28d: bool = True, save_path: str = None):
    """为某个体画 Normal / Acute / Chronic 的1秒波形（同一通道）"""
    all_files, _ = list_h5_by_group_and_subject_root(DATA_DIR, SELECTED_SUBJECTS)
    groups = group_files_by_subject_then_state(all_files, labels_csv=CSV_PATH)
    sd = groups.get(subject, None)
    if not sd:
        print(f"[Plot] 找不到个体 {subject}"); return

    pick_norm  = sd.get("状态1_建模前", [])
    pick_acute = sd.get("状态2_IV级前", []) or sd.get("状态3_止惊前后", [])
    pick_chron = sd.get("状态4_建模后", [])

    def _pick_chronic(lst):
        if not lst: return None
        if prefer_28d:
            for fp in lst:
                if "28" in os.path.basename(fp): return fp
        return lst[-1]  

    fp_N = pick_norm[0]  if pick_norm else None
    fp_A = pick_acute[0] if pick_acute else None
    fp_C = _pick_chronic(pick_chron)

    labels = [("Normal", fp_N), ("Acute", fp_A), ("Chronic", fp_C)]
    labels = [(name, fp) for name, fp in labels if fp is not None]
    if not labels:
        print(f"[Plot] {subject} 没有足够阶段数据"); return
    segs = []
    for name, fp in labels:
        t_ms, seg, Fs = _read_1s_segment(fp, ch_idx, sec=sec, where="middle")
        segs.append((t_ms, seg, Fs))
    global_min = min(np.percentile(seg, 1) for (_, seg, _) in segs)
    global_max = max(np.percentile(seg, 99) for (_, seg, _)  in segs)

    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 6), sharex=True)
    if len(labels) == 1: axes = [axes]
    for ax, (name, fp), (t_ms, seg, Fs) in zip(axes, labels, segs):
        # t_ms, seg, Fs = _read_1s_segment(fp, ch_idx, sec=sec, where="middle")
        # t_ms = np.arange(seg.size) / Fs * 1000.0
        ax.plot(t_ms, seg, lw=0.9)
        # ax.set_ylim(np.percentile(seg, [1, 99])[0], np.percentile(seg, [1, 99])[1])
        ax.set_ylim(global_min, global_max)
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(f"{subject} - {name} | {os.path.basename(fp)} | ch={ch_idx}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"[Saved] {save_path}")
    plt.show()


# ==============================
# Run probability sliding windows on the full concatenated signal.
# ==============================

def predict_probs_on_signal(sig_norm: np.ndarray, Fs: float,
                            win_sec: float, stride_sec: float) -> tuple:
    """
    sig_norm: 归一化后的整段 [C, T]
    返回 (times[N], probs[N])，与你现有 predict_probs_on_file 的输出形式一致（p=class1的概率）。
    """
    model = predict_probs_on_signal._model  
    device = predict_probs_on_signal._device

    C, N = sig_norm.shape
    win = int(win_sec*Fs); hop = int(stride_sec*Fs)
    if win <= 0 or N < win:
        return np.array([]), np.array([])
    starts = list(range(0, N - win + 1, hop))

    probs, times = [], []
    model.eval()
    for s in starts:
        seg = sig_norm[:, s:s+win].astype(np.float32)
        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0]           # [N_CLASSES]
            p = prob[CPD_POS_CLASS if prob.numel()>CPD_POS_CLASS else -1].item()
            # p = torch.softmax(model(x), dim=1)[:, 1].item()
        probs.append(p)
        times.append((s + win/2)/Fs)
    return np.asarray(times), np.asarray(probs)

# Closure-bound model and device assigned by the main pipeline.
predict_probs_on_signal._model = None
predict_probs_on_signal._device = DEVICE

def predict_feats_on_signal(sig_norm: np.ndarray, Fs: float,
                            win_sec: float, stride_sec: float) -> tuple:
    """
    与 predict_probs_on_signal 对称：返回 (times[N], feats[N, D])
    """
    model = predict_probs_on_signal._model
    device = predict_probs_on_signal._device

    C, N = sig_norm.shape
    win = int(win_sec*Fs); hop = int(stride_sec*Fs)
    if win <= 0 or N < win:
        return np.array([]), np.zeros((0, 256), dtype=np.float32)
    starts = list(range(0, N - win + 1, hop))

    feats, times = [], []
    model.eval()
    for s in starts:
        seg = sig_norm[:, s:s+win].astype(np.float32)
        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            fvec = model.forward_features(x).detach().cpu().numpy()[0]  # [256]
        feats.append(fvec)
        times.append((s + win/2)/Fs)
    return np.asarray(times), np.asarray(feats, dtype=np.float32)


def _run_cpd_on_concat(model, file_seq, title_prefix, prob_png, heat_png):
    """小工具：给定一段文件序列 -> 拼接 -> 推理 -> ruptures CPD -> 画 1 张概率图 + 1 张热图"""
    try:
        sig_cat, Fs, bounds = concat_files_in_order(file_seq, ds_step=DS_STEP)
    except Exception as e:
        print(f"[{title_prefix}] 拼接失败 — {e}")
        return []

    ok, info = check_concat_validity(file_seq, sig_cat, Fs, bounds)
    print(f"[{title_prefix}] files={info['n_files']} | dur={info['dur_sec']:.1f}s | Fs={info['Fs']:.1f} | notes={info['notes']}")
    if not ok:
        return []

    # Probability and feature sliding-window section.
    times, probs = predict_probs_on_signal(sig_cat, Fs, WIN_SEC, STRIDE_SEC)
    if len(times) == 0:
        print(f"[{title_prefix}] 序列太短，无法滑窗。")
        return []

    k_prob = max(1, int(round(SMOOTH_SEC / max(STRIDE_SEC, 1e-6))))
    p_smooth = moving_average(probs, k_prob)
    TH_ENTER, TH_EXIT = TH_HIGH_FIXED, TH_LOW_FIXED

    # Align features with times.
    times_f, feats = predict_feats_on_signal(sig_cat, Fs, WIN_SEC, STRIDE_SEC)
    if len(times_f) != len(times) or (len(times_f) and not np.allclose(times_f, times)):
        t_left, t_right = times_f[0], times_f[-1]
        from scipy.interpolate import interp1d
        F = interp1d(times_f, feats, axis=0, kind="linear", bounds_error=False,
                     fill_value=(feats[0], feats[-1]))
        feats = F(np.clip(times, t_left, t_right))

    # Run ruptures in feature space.
    cps_feat = cpd_from_embeddings(
        times, feats,
        smooth_sec=0.8,
        stride_sec=STRIDE_SEC,
        pca_dim=10,
        method="pelt-rbf",
        pen=7.5,
        min_sep=1.0
    )

    # Apply refractory period, boundary removal, and significance filtering using PC1.
    cps_lock = _apply_refractory(cps_feat, refract_sec=4.0)
    cps_lock = remove_cps_near_bounds(cps_lock, bounds, pad_sec=3.0)

    # Use PC1 as the reference sequence for significance filtering.
    X = feats.astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(Xc, full_matrices=False)
        series_1d = (Xc @ vh[0]).astype(np.float32)
    except Exception:
        series_1d = Xc[:, 0].astype(np.float32)

    cps_final = keep_if_significant_ttest(
        cps_lock, series_1d, times,
        pre=3.0, post=3.0, alpha=0.05, min_pts=3, min_effect=0.10
    )

    # Draw one summary figure for each output type.
    plot_prob_with_cps(
        times, p_smooth, TH_ENTER, TH_EXIT, cps_final,
        bounds=bounds, title=title_prefix + " · probability curve",
        save_path=prob_png
    )
#     plot_signal_with_cps(
#     sig_cat, Fs, cps_final,
# Change this channel index to inspect another channel.
#     bounds=bounds,
# Plot title for the raw signal curve.
# The output filename can be changed, for example to *_sig.png.
# )
    plot_channel_energy_heatmap(
        sig_cat, Fs, cps_final,
        title=title_prefix + " · energy heatmap across 8 channels",
        save_path=heat_png
    )

    cp_str = ", ".join([f"{t:.2f}" for t in cps_final])
    print(f"[{title_prefix}] CP={len(cps_final)} | {cp_str}")
    return cps_final


def cpd_on_subject_two_merged_blocks(model, files_all: list, save_txt="cp_times_two_blocks.txt"):
    """
    把每只动物的数据合并为两段：
      Block A = 状态1_建模前 + 状态2_IV级前（保持内部时间顺序）
      Block B = 状态3_止惊前后 + 状态4_建模后（保持内部时间顺序）
    每段只输出一张概率图 + 一张热图。
    """
    groups = group_files_by_subject_then_state(files_all, labels_csv=CSV_PATH)
    predict_probs_on_signal._model = model
    predict_probs_on_signal._device = DEVICE

    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            f.write("# subject\tblock\tchangepoint_times_seconds\n")

    outdir = os.path.join("cpd_plots", "two_blocks")
    os.makedirs(outdir, exist_ok=True)

    for subject, state_dict in groups.items():
        # Preserve the original state_dict order after previous time/CSV sorting.
        blockA = state_dict.get("状态1_建模前", []) + state_dict.get("状态2_IV级前", [])
        blockB = state_dict.get("状态3_止惊前后", []) + state_dict.get("状态4_建模后", [])
        if not blockA and not blockB:
            continue

        print(f"\n====== [Subject] {subject or '(unknown)'}  (A:{len(blockA)} files, B:{len(blockB)} files) ======")

        # Segment A.
        if blockA:
            titleA = f"{subject} · Pre-Modeling+Pre IV level"
            probA  = os.path.join(outdir, f"{_safe_name(subject)}__A_prob.png")
            heatA  = os.path.join(outdir, f"{_safe_name(subject)}__A_heat.png")
            cpsA = _run_cpd_on_concat(model, blockA, titleA, probA, heatA)
            if save_txt:
                with open(save_txt, "a", encoding="utf-8") as f:
                    f.write(f"{subject}\tA\t" + ",".join([f"{t:.3f}" for t in cpsA]) + "\n")

        # Segment B.
        if blockB:
            titleB = f"{subject} · Pre and Post Seizures+Post-Modeling"
            probB  = os.path.join(outdir, f"{_safe_name(subject)}__B_prob.png")
            heatB  = os.path.join(outdir, f"{_safe_name(subject)}__B_heat.png")
            cpsB = _run_cpd_on_concat(model, blockB, titleB, probB, heatB)
            if save_txt:
                with open(save_txt, "a", encoding="utf-8") as f:
                    f.write(f"{subject}\tB\t" + ",".join([f"{t:.3f}" for t in cpsB]) + "\n")

def _run_cpd_on_concat_with_spans(model, file_seq, spans, title_prefix, prob_png, heat_png):
    """
    与 _run_cpd_on_concat 类似，但额外叠加 spans（如 A/B 段）底色区间。
    spans: list of (name, t0, t1, color, alpha)
    """
    try:
        sig_cat, Fs, bounds = concat_files_in_order(file_seq, ds_step=DS_STEP)
    except Exception as e:
        print(f"[{title_prefix}] 拼接失败 — {e}")
        return []

    ok, info = check_concat_validity(file_seq, sig_cat, Fs, bounds)
    print(f"[{title_prefix}] files={info['n_files']} | dur={info['dur_sec']:.1f}s | Fs={info['Fs']:.1f} | notes={info['notes']}")
    if not ok:
        return []

    # Probability and feature extraction.
    times, probs = predict_probs_on_signal(sig_cat, Fs, WIN_SEC, STRIDE_SEC)
    if len(times) == 0:
        print(f"[{title_prefix}] 序列太短，无法滑窗。")
        return []
    k_prob = max(1, int(round(SMOOTH_SEC / max(STRIDE_SEC, 1e-6))))
    p_smooth = moving_average(probs, k_prob)
    TH_ENTER, TH_EXIT = TH_HIGH_FIXED, TH_LOW_FIXED

    times_f, feats = predict_feats_on_signal(sig_cat, Fs, WIN_SEC, STRIDE_SEC)
    if len(times_f) != len(times) or (len(times_f) and not np.allclose(times_f, times)):
        t_left, t_right = times_f[0], times_f[-1]
        from scipy.interpolate import interp1d
        F = interp1d(times_f, feats, axis=0, kind="linear", bounds_error=False,
                     fill_value=(feats[0], feats[-1]))
        feats = F(np.clip(times, t_left, t_right))

    cps_feat = cpd_from_embeddings(
        times, feats, smooth_sec=0.8, stride_sec=STRIDE_SEC,
        pca_dim=10, method="pelt-rbf", pen=7.5, min_sep=1.0
    )
    cps_lock = _apply_refractory(cps_feat, refract_sec=4.0)
    cps_lock = remove_cps_near_bounds(cps_lock, bounds, pad_sec=3.0)

    # Use PC1 for significance filtering.
    X = feats.astype(np.float32); Xc = X - X.mean(0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(Xc, full_matrices=False)
        series_1d = (Xc @ vh[0]).astype(np.float32)
    except Exception:
        series_1d = Xc[:, 0].astype(np.float32)

    cps_final = keep_if_significant_ttest(
        cps_lock, series_1d, times, pre=3.0, post=3.0, alpha=0.05, min_pts=3, min_effect=0.10
    )

    # Overlay file-span annotations on the plot.
    plot_prob_with_cps(
        times, p_smooth, TH_ENTER, TH_EXIT, cps_final,
        bounds=bounds, title=title_prefix + " · full-time probability curve",
        save_path=prob_png, spans=spans
    )

    plot_channel_energy_heatmap(
        sig_cat, Fs, cps_final, title=title_prefix + " · full-time energy heatmap across 8 channels",
        save_path=heat_png, spans=spans
    )
    print(f"[{title_prefix}] CP={len(cps_final)} | " + ", ".join(f"{t:.2f}" for t in cps_final))
    return cps_final


def cpd_on_subject_full_timeline(model, files_all: list, save_txt="cp_times_full.txt"):
    """
    每只动物把 A 段(建模前+IV级前) 与 B 段(止惊前后+建模后)按顺序合起来，
    输出一张“全程”概率图 + 一张“全程”热图。
    """
    groups = group_files_by_subject_then_state(files_all, labels_csv=CSV_PATH)
    predict_probs_on_signal._model = model
    predict_probs_on_signal._device = DEVICE

    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            f.write("# subject\tchangepoint_times_seconds\n")

    outdir = os.path.join("cpd_plots", "full_timeline")
    os.makedirs(outdir, exist_ok=True)

    for subject, state_dict in groups.items():
        blockA = state_dict.get("状态1_建模前", []) + state_dict.get("状态2_IV级前", [])
        blockB = state_dict.get("状态3_止惊前后", []) + state_dict.get("状态4_建模后", [])
        file_seq = blockA + blockB
        if not file_seq:
            continue

        # Run one concatenation pass to obtain per-file boundaries for A/B intervals.
        sig_tmp, Fs_tmp, bounds = concat_files_in_order(file_seq, ds_step=DS_STEP)
        if sig_tmp is None or not bounds:
            continue

        # CPD pipeline and subject-level visualization note.
        if len(blockA) > 0:
            a_start = bounds[0][1]
            a_end   = bounds[len(blockA)-1][2]
            spans = [("Block A", a_start, a_end, "tab:blue", 0.10)]
            b_start = a_end
            b_end   = bounds[-1][2]
            if len(blockB) > 0:
                spans.append(("Block B", b_start, b_end, "tab:orange", 0.10))
        else:
            # CPD pipeline and subject-level visualization note.
            a_start = bounds[0][1]
            a_end   = bounds[0][1]
            b_start = a_start; b_end = bounds[-1][2]
            spans = [("Block B", b_start, b_end, "tab:orange", 0.10)]

        title = f"{subject} · full-time (Pre-Modeling+Pre IV level, B:止惊前后+建模后)"
        prob_png = os.path.join(outdir, f"{_safe_name(subject)}__FULL_prob.png")
        heat_png = os.path.join(outdir, f"{_safe_name(subject)}__FULL_heat.png")

        cps = _run_cpd_on_concat_with_spans(model, file_seq, spans, title, prob_png, heat_png)

        if save_txt:
            with open(save_txt, "a", encoding="utf-8") as f:
                f.write(f"{subject}\t" + ",".join(f"{t:.3f}" for t in cps) + "\n")


# CPD pipeline and subject-level visualization note.
def select_files_balanced_from_subset(h5_files, total_target=6, min_per_class=2):
    return list(h5_files)

class SubsetSequentialSampler(Sampler):
    """Yields given indices in ascending order (no shuffle)."""
    def __init__(self, indices):
        self.indices = list(sorted(map(int, indices)))
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def chronological_split_by_file(ds, val_ratio=0.2):
    """
    按“每个文件内部”的时间顺序切分。
    前段 -> 训练；后段 -> 验证。
    返回: train_idx, val_idx (np.ndarray[int])
    """
    from collections import defaultdict
    groups = defaultdict(list)  # Dataset construction and window-indexing note.
    for i, (fp, _, _) in enumerate(ds.index_list):
        groups[fp].append(i)
    train_idx, val_idx = [], []
    for fp, idxs in groups.items():
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs);continue
        cut = max(1, int(round((1.0 - val_ratio) * n)))
        cut = min(cut, n - 1)
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int)


def debug_check_csv_labels(files, max_show=10):
    global LABEL_MAP
    if LABEL_MAP is None:
        LABEL_MAP = _load_labels_from_csv(CSV_PATH)
    print(f"[CSV] Entries: {len(LABEL_MAP)} | Sample path: {CSV_PATH}")
    shown = 0
    mismatch = 0
    for fp in files:
        base = os.path.basename(fp)
        csv_val = LABEL_MAP.get(_norm_name(base), LABEL_MAP.get(os.path.splitext(_norm_name(base))[0], None))
        h5_val  = read_label_h5only(fp)
        final   = read_label(fp)
        used = "CSV" if csv_val is not None else ("SKIP" if final < 0 else "H5")
        if shown < max_show:
            print(f"  {base}  ->  CSV:{csv_val}  H5:{h5_val}  USED:{final} ({used})")
            shown += 1
        if csv_val is not None and final != csv_val:
            mismatch += 1
    if mismatch>0:
        print(f"[CSV] WARNING: {mismatch} file(s) not using CSV value as final.")

# Dataset construction and window-indexing note.
class EEGWindowDataset(Dataset):
    def __init__(self, files, win_sec=1.0, stride_sec=4.0,
                 split='train', train_stride_factor=1,
                 max_windows_per_file=80, ds_step=1, seed=42,
                 # Dataset construction and window-indexing note.
                 class_pre_ignore=None,
                 class_post_ignore=None):
        self.index_list, self.labels = [], []
        self.win_sec, self.stride_sec = float(win_sec), float(stride_sec)
        self.ds_step = max(1, int(ds_step))
        self.split = str(split).lower()
        self.train_stride_factor = max(1, int(train_stride_factor))
        self.file_stats = {}

        # Dataset construction and window-indexing note.
        self.class_pre_ignore = dict(class_pre_ignore or {})
        self.class_post_ignore = dict(class_post_ignore or {})

        rng = np.random.default_rng(seed)

        for fp in files:
            try:
                with h5py.File(fp, "r") as f:
                    # C, N = f["/sig"].shape
                    N = f["/sig"].shape[1]
                    Fs = safe_item(f["/sig"].attrs["Fs"])
            except Exception as e:
                print(f"[WARN] 读取失败，跳过: {os.path.basename(fp)} — {e}")
                continue

            # Dataset construction and window-indexing note.
            mu, mad = get_file_stats(fp, stat_hz=10)
            # self.file_stats[fp] = (mu, mad)
            # self.file_stats[fp] = (mu[KEEP_CH_IDX, :], mad[KEEP_CH_IDX, :])
            self.file_stats[fp] = (mu, mad)

            win = int(self.win_sec * Fs)
            stride = int(self.stride_sec * Fs)
            if win <= 0 or N - win + 1 <= 0:
                continue

            # Dataset construction and window-indexing note.
            starts = list(range(0, N - win + 1, stride))

            # Limit the number of windows per file.
            if self.split == 'train' and self.train_stride_factor > 1:
                starts = starts[::self.train_stride_factor]

            # Dataset construction and window-indexing note.
            if max_windows_per_file and len(starts) > max_windows_per_file:
                step = max(1, len(starts) // max_windows_per_file)
                starts = starts[::step][:max_windows_per_file]

            # Dataset construction and window-indexing note.
            y = read_label(fp)
            if y < 0:
                # Dataset construction and window-indexing note.
                continue

            # Convert seconds to a number of windows.
            if self.split == 'train':
                pre_cut = float(self.class_pre_ignore.get(int(y), 0.0))
                post_cut = float(self.class_post_ignore.get(int(y), 0.0))
                # Dataset construction and window-indexing note.
                pre_drop = int(round(pre_cut / max(self.stride_sec, 1e-6)))
                post_drop = int(round(post_cut / max(self.stride_sec, 1e-6)))
                if pre_drop > 0:
                    starts = starts[min(pre_drop, len(starts)):]  # Dataset construction and window-indexing note.
                if post_drop > 0 and len(starts) > post_drop:
                    starts = starts[:-post_drop]  # Dataset construction and window-indexing note.
                # Add all retained windows to the index for train/validation/test use.
                if not starts:
                    continue

            # Dataset construction and window-indexing note.
            for s in starts:
                self.index_list.append((fp, s, Fs))
                self.labels.append(int(y))

        if not self.index_list:
            raise RuntimeError("No windows found in subset. 放宽限制或检查数据/标签/屏蔽参数。")

        print(f"[Dataset] windows: {len(self.index_list)} | files used: {len(files)} | split={self.split}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        fp, start, Fs = self.index_list[idx]
        win = int(self.win_sec * Fs)
        with h5py.File(fp, "r") as f: 
            # seg = f["/sig"][KEEP_CH_IDX, start:start+win]   # (C, win)
            seg = f["/sig"][CHANNEL_SLICE, start:start+win]
        if self.ds_step > 1:
            seg = seg[:, ::self.ds_step]
        mu, mad = self.file_stats.get(fp, get_file_stats(fp, stat_hz=10))
        seg = np.clip((seg - mu) / mad, -5, 5).astype(np.float32)
        return torch.from_numpy(seg), torch.tensor(self.labels[idx], dtype=torch.long)


class AttnPool(nn.Module):
    def __init__(self, d, tau=1.0):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        self.tau = float(tau)

    def forward(self, x, mask: torch.Tensor = None, return_weights: bool = False):
        # raw scores: [B, T, 1] -> [B, T]
        s = self.attn(x).squeeze(-1)

        # Neural network architecture note.
        s = s - s.max(dim=1, keepdim=True).values
        if self.tau != 1.0:
            s = s / self.tau

        if mask is not None:
            s = s.masked_fill(mask == 0, float('-inf'))

        a = torch.softmax(s, dim=1)  # [B, T]

        pooled = (x * a.unsqueeze(-1)).sum(dim=1)  # [B, D]
        if return_weights:
            return pooled, a
        return pooled


# Neural network architecture note.
# Model: CNN + BiLSTM classifier with attention pooling.
class EEG_CNN_LSTM(nn.Module):
    def __init__(self, in_ch, n_classes=2, lstm_hidden=64, lstm_layers=1, lstm_dropout=0.3):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), # First convolution layer; original kernel was 11 with padding 5.
                                nn.MaxPool1d(2), nn.Dropout(0.4))  # Dropout was originally 0.2.
        self.b2 = nn.Sequential(nn.Conv1d(32, 64, 10, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.MaxPool1d(2), nn.Dropout(0.4))
        self.b3 = nn.Sequential(nn.Conv1d(64,128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
                                nn.MaxPool1d(2), nn.Dropout(0.4))
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=(lstm_dropout if lstm_layers > 1 else 0.0))
        self.post_lstm_dropout = nn.Dropout(lstm_dropout)  # Neural network architecture note.
        self.pool = AttnPool(2*lstm_hidden, tau=1.0)  # Neural network architecture note.
        self.fc1  = nn.Linear(2*lstm_hidden, 256)  # Lightweight regularization.
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, n_classes))
    
    def _features_with_attn(self, x, mask=None, return_attn=False):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)   # [B, 128, T']
        x = x.permute(0, 2, 1)                           # [B, T', 128]
        out, _ = self.lstm(x)                            # [B, T', 2H]
        out = self.post_lstm_dropout(out)
        if return_attn:
            pooled, attn = self.pool(out, mask=mask, return_weights=True)  # [B, 2H], [B, T']
            feat = self.fc1(pooled)                                        # [B, 256]
            return feat, attn
        else:
            pooled = self.pool(out, mask=mask, return_weights=False)       # [B, 2H]
            feat = self.fc1(pooled)                                        # [B, 256]
            return feat
    
    def forward(self, x, mask=None):           # [B,C,T]
        feat = self._features_with_attn(x, mask=mask, return_attn=False)
        return self.head(feat)      # logits
    @torch.no_grad()
    def forward_features(self, x):  # Return embeddings with shape [B, 256].
        self.eval()
        return self._features_with_attn(x, mask=None, return_attn=False)

# Neural network architecture note.

def predict_probs_on_file(fp, model, win_sec, stride_sec, ds_step=1, device="cpu"):
    model.eval()
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :]) 
        # sig = np.array(f["/sig"])   # (C, N)
        Fs  = safe_item(f["/sig"].attrs["Fs"])
    # Use only the first eight channels.
    # seg = sig[:, s:s+win]
    C, N = sig.shape
    win = int(win_sec*Fs); stride = int(stride_sec*Fs)
    if win <= 0 or N < win:
        return np.array([]), np.array([]), Fs
    starts = list(range(0, N - win + 1, stride))
    probs, times = [], []
    for s in starts:
        seg = sig[:, s:s+win]
        if ds_step>1: seg = seg[:, ::ds_step]
        # === PATCH-RN-INFER: normalize each window using per-file stats ===
        mu, mad = get_file_stats(fp, stat_hz=10)  # Prediction, embedding, or audit helper note.
        seg = np.clip((seg - mu) / mad, -5, 5).astype(np.float32)

        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            # p = torch.softmax(model(x), dim=1)[0,1].item()
            prob = torch.softmax(model(x), dim=1)[0]
            p = prob[CPD_POS_CLASS if prob.numel()>CPD_POS_CLASS else -1].item()
        probs.append(p)
        center = (s + win/2.0) / Fs
        times.append(center)
    return np.array(times), np.array(probs), Fs

def predict_probs_on_array(sig, Fs, model, win_sec, stride_sec, ds_step=1, device="cpu", fp_for_stats=None):
    model.eval()
    C, N = sig.shape
    win = int(win_sec*Fs); stride = int(stride_sec*Fs)
    if win <= 0 or N < win:
        return np.array([]), np.array([]), Fs
    starts = list(range(0, N - win + 1, stride))
    probs, times = [], []

    if fp_for_stats is not None:
        mu, mad = get_file_stats(fp_for_stats, stat_hz=10)
    else:
        mu = np.median(sig, axis=1, keepdims=True)
        mad = np.median(np.abs(sig - mu), axis=1, keepdims=True) + 1e-6
    for s in starts:
        seg = sig[:, s:s+win]
        if ds_step > 1:
            seg = seg[:, ::ds_step]
        seg = np.clip((seg - mu) / mad, -5, 5).astype(np.float32)

        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0]
            p = prob[CPD_POS_CLASS if prob.numel() > CPD_POS_CLASS else -1].item()
        probs.append(p)
        center =  (s + win/2.0) / Fs
        times.append(center)
    return np.array(times), np.array(probs), Fs

def _pearson_corr(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.size < 3 or b.size != a.size: 
        return float('nan')
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))
def _norm_in_memory(sig: np.ndarray, Fs: float, stat_hz: float = 10.0, eps: float = 1e-4):
    """
    用与你训练/推理一致的 robust 归一化逻辑，但对“内存数组”执行（不依赖 fp）。
    """
    step = max(1, int(round(Fs / float(stat_hz))))
    sig_dec = sig[:, ::step]
    mu  = np.median(sig_dec, axis=1, keepdims=True).astype(np.float32)
    mad = np.median(np.abs(sig_dec - mu), axis=1, keepdims=True).astype(np.float32)
    mad = np.maximum(mad, eps)
    x = np.clip((sig - mu) / mad, -5, 5).astype(np.float32)
    return x
def _predict_on_array(sig_arr: np.ndarray, Fs: float, model):
    """
    对内存中的标准化前的 raw 信号做一次完整推理（含 robust 归一化、滑窗、softmax）。
    依赖你的 predict_probs_on_signal（其内部需要先绑定 _model/_device）。
    """
    # Prediction, embedding, or audit helper note.
    X = _norm_in_memory(sig_arr, Fs, stat_hz=10.0)
    # Prediction, embedding, or audit helper note.
    predict_probs_on_signal._model = model
    predict_probs_on_signal._device = DEVICE
    t, p = predict_probs_on_signal(X, Fs, WIN_SEC, STRIDE_SEC)
    return t, p
@torch.no_grad()
def audit_invariance(fp: str, model) -> dict:
    """
    在单个 .h5 文件上做：
      - 振幅缩放 0.5x / 2x（应基本不变，因为有 per-file robust norm）
      - 时间平移 1 个 stride（曲线应相对对齐）
      - 时间反转（不应系统性“更高”）
    返回各指标字典。
    """
    # Prediction, embedding, or audit helper note.
    t0, p0, Fs = predict_probs_on_file(fp, model, WIN_SEC, STRIDE_SEC, DS_STEP, DEVICE)

    # Prediction, embedding, or audit helper note.
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])
        Fs  = safe_item(f["/sig"].attrs["Fs"])

    # Prediction, embedding, or audit helper note.
    t1, p1 = _predict_on_array(sig * 0.5, Fs, model)
    t2, p2 = _predict_on_array(sig * 2.0,  Fs, model)

    # Prediction, embedding, or audit helper note.
    hop = int(STRIDE_SEC * Fs)
    sig_shift = np.pad(sig, ((0, 0), (hop, 0)), mode="constant")[:, :sig.shape[1]]
    ts, ps = _predict_on_array(sig_shift, Fs, model)

    # Time reversal.
    tr, pr = _predict_on_array(sig[:, ::-1].copy(), Fs, model)

    # Prediction, embedding, or audit helper note.
    amp_corr_05 = _pearson_corr(p0, p1)
    amp_corr_20 = _pearson_corr(p0, p2)
    shift_corr  = _pearson_corr(p0[:-1], ps[1:]) if len(p0) > 1 and len(ps) > 1 else float('nan')

    report = {
        "file": os.path.basename(fp),
        "amp_corr_0.5x": float(amp_corr_05),
        "amp_corr_2x":   float(amp_corr_20),
        "shift_corr":    float(shift_corr),
        "orig_mean":     float(np.mean(p0)) if len(p0) else float('nan'),
        "rev_mean":      float(np.mean(pr)) if len(pr) else float('nan'),
        "n_points":      int(len(p0)),
    }
    return report

def run_audit_over_dir(data_dir: str, model, pattern: str = "**/*.h5", out_csv: str = "audit_invariance.csv"):
    """
    递归扫描 data_dir 下所有 .h5 文件，对每个文件跑 audit_invariance，
    最后把结果写入 CSV（逗号分隔，UTF-8）。
    """
    import csv
    from glob import glob
    files = sorted(glob(os.path.join(data_dir, pattern), recursive=True))
    if not files:
        print(f"[Audit] No .h5 found under: {data_dir}")
        return

    print(f"[Audit] Found {len(files)} files. Running invariance checks...")
    rows = []
    for i, fp in enumerate(files, 1):
        try:
            rep = audit_invariance(fp, model)
        except Exception as e:
            rep = {"file": os.path.basename(fp), "error": str(e)}
        rows.append(rep)
        if i % 5 == 0 or i == len(files):
            print(f"  - {i}/{len(files)} done")

    # Prediction, embedding, or audit helper note.
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Audit] Saved: {out_csv}")

    print("\n[Audit] Quick summary:")
    for r in rows:
        if "error" in r:
            print(f"  {r['file']}  ERROR: {r['error']}")
        else:
            print(f"  {r['file']}: amp_corr_0.5x={r['amp_corr_0.5x']:.3f}, "
                  f"amp_corr_2x={r['amp_corr_2x']:.3f}, "
                  f"shift_corr={r['shift_corr']:.3f}, "
                  f"orig_mean={r['orig_mean']:.3f}, rev_mean={r['rev_mean']:.3f}")

def predict_feats_on_file(fp, model, win_sec, stride_sec, ds_step=1, device="cpu"):
    model.eval()
    with h5py.File(fp, "r") as f:
        # sig = np.array(f["/sig"])   # (C, N)
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])
        Fs  = safe_item(f["/sig"].attrs["Fs"])
        # Fs  = safe_item(f["/sig"].attrs["Fs"])
    # sig = sig[KEEP_CH_IDX, :]
    # sig = np.array(f["/sig"][CHANNEL_SLICE, :]) 
    C, N = sig.shape
    win = int(win_sec*Fs); stride = int(stride_sec*Fs)
    if win <= 0 or N < win:
        return np.array([]), np.zeros((0,256), dtype=np.float32), Fs
    starts = list(range(0, N - win + 1, stride))
    feats, times = [], []
    for s in starts:
        seg = sig[:, s:s+win]
        if ds_step>1: seg = seg[:, ::ds_step]
        # === PATCH-RN-INFER: normalize each window using per-file stats ===
        mu, mad = get_file_stats(fp, stat_hz=10)  # Prediction, embedding, or audit helper note.
        seg = np.clip((seg - mu) / mad, -5, 5).astype(np.float32)

        x = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            fvec = model.forward_features(x).detach().cpu().numpy()[0]
        feats.append(fvec)
        center = (s + win/2.0) / Fs
        times.append(center)
    return np.array(times), np.array(feats, dtype=np.float32), Fs

# Moving average.
def moving_average(x, k):
    if k <= 1: return x
    k = int(max(1, k))
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    res = (cumsum[k:] - cumsum[:-k]) / float(k)
    pad_left = np.full(k//2, res[0] if len(res)>0 else x[0] if len(x)>0 else 0.0)
    pad_right= np.full(k - k//2 - 1, res[-1] if len(res)>0 else x[-1] if len(x)>0 else 0.0)
    return np.concatenate([pad_left, res, pad_right])

# def hysteresis_binarize(p, th_high=0.7, th_low=0.3):
#     state = np.zeros_like(p, dtype=int)
#     curr = 0
#     for i, val in enumerate(p):
#         if curr == 0 and val >= th_high:
#             curr = 1
#         elif curr == 1 and val <= th_low:
#             curr = 0
#         state[i] = curr
#     return state

# def enforce_min_duration(state, times, min_state_sec=3.0, min_gap_sec=1.0):
#     if len(state)==0:
#         return state.copy()
#     s = state.copy()
#     runs = []
#     start = 0
#     for i in range(1,len(s)):
#         if s[i] != s[i-1]:
#             runs.append((start, i-1, s[i-1]))
#             start = i
#     runs.append((start, len(s)-1, s[-1]))
#     changed = True
#     while changed and len(runs)>1:
#         changed = False
#         new_runs = []
#         i = 0
#         while i < len(runs):
#             a0,a1,lab = runs[i]
#             dur = times[a1] - times[a0]
#             if dur < min_state_sec and len(runs)>1:
#                 if i==0:
#                     b0,b1,lab2 = runs[i+1]
#                     new_runs.append((a0,b1,lab2))
#                     i += 2; changed=True
#                 elif i==len(runs)-1:
#                     p0,p1,lab2 = new_runs[-1]
#                     new_runs[-1] = (p0, a1, lab2)
#                     i += 1; changed=True
#                 else:
#                     p0,p1,labL = new_runs[-1]
#                     n0,n1,labR = runs[i+1]
#                     gapL = times[a0] - times[p1]
#                     gapR = times[n0] - times[a1]
#                     if gapL <= gapR:
#                         new_runs[-1] = (p0, a1, labL)
#                         i += 1; changed=True
#                     else:
#                         new_runs.append((a0, n1, labR))
#                         i += 2; changed=True
#             else:
#                 new_runs.append(runs[i]); i += 1
#         runs = new_runs
#     merged = []
#     i=0
#     while i < len(runs):
#         if i < len(runs)-2:
#             a0,a1,labA = runs[i]
#             g0,g1,labG = runs[i+1]
#             b0,b1,labB = runs[i+2]
#             if labA==labB and (times[g1]-times[g0]) < min_gap_sec:
#                 merged.append((a0,b1,labA))
#                 i += 3
#                 continue
#         merged.append(runs[i]); i+=1
#     out = s.copy()
#     for a0,a1,lab in merged:
#         out[a0:a1+1] = lab
#     return out

# def states_to_changepoints(state, times):
#     cps = []
#     for i in range(1,len(state)):
#         if state[i] != state[i-1]:
#             cps.append( (times[i-1] + times[i]) / 2.0 )
#     return cps

def _apply_refractory(cps, refract_sec):
    """触发后不应期"""
    if not cps or refract_sec <= 0:
        return cps
    out, last = [], -1e9
    for t in sorted(float(x) for x in cps):
        if t - last >= refract_sec:
            out.append(t); last = t
    return out

def _dedup_by_min_sep(ts, min_sep):
    """按最小间隔去重（秒）"""
    ts = sorted(float(t) for t in ts)
    out = []
    for t in ts:
        if not out or (t - out[-1]) >= float(min_sep):
            out.append(t)
    return out

def cpd_from_embeddings(times, feats,
                        smooth_sec=1.0,
                        stride_sec=1.0,
                        pca_dim=8,
                        method="pelt-rbf",
                        pen=10.0,
                        min_sep=1.0):
    """
    只基于 ruptures 在特征空间做 CPD，不再回退到 CUSUM。
    """
    times = np.asarray(times)
    if feats is None or len(times) == 0 or len(feats) != len(times):
        return []

    X = np.asarray(feats, dtype=np.float32)  # [N, D]

    # Smooth each feature dimension with a moving average.
    k = max(1, int(round(smooth_sec / max(stride_sec, 1e-6))))
    if k > 1:
        Xs = np.vstack([moving_average(X[:, i], k)
                        for i in range(X.shape[1])]).T
    else:
        Xs = X

    # Feature-space change-point detection note.
    d = int(min(pca_dim, Xs.shape[1]))
    if d >= 1:
        try:
            Z = PCA(n_components=d,
                    svd_solver="auto",
                    random_state=0).fit_transform(Xs)
        except Exception:
            Z = Xs  # If PCA fails, use the original features.
    else:
        Z = Xs

    # Use only ruptures PELT; no CUSUM fallback.
    if not _HAS_RUPTURES or not method.lower().startswith("pelt"):
        raise RuntimeError(
            "cpd_from_embeddings 现在只支持 ruptures PELT，"
            "请确保已安装 ruptures 并设置 method='pelt-rbf' 或 'pelt-l2'。"
        )

    # Feature-space change-point detection note.
    model_name = "rbf" if "rbf" in method.lower() else "l2"
    algo = rpt.Pelt(model=model_name).fit(Z)

    # Feature-space change-point detection note.
    _pen = float(pen)
    cp_idx = algo.predict(pen=_pen)  # Feature-space change-point detection note.

    # Feature-space change-point detection note.
    if cp_idx and cp_idx[-1] == len(Z):
        cp_idx = cp_idx[:-1]

    cp_times = [times[i-1] for i in cp_idx if 1 <= i < len(times)]

    # Feature-space change-point detection note.
    cp_times = _dedup_by_min_sep(cp_times, min_sep=min_sep)
    return cp_times


def remove_cps_near_bounds(cps, bounds, pad_sec=8.0):
    if not bounds or not cps:
        return cps
    taboo = []
    for _, a, b in bounds:
        taboo.extend([float(a), float(b)])
    out = []
    for t in cps:
        if all(abs(float(t) - tb) > pad_sec for tb in taboo):
            out.append(float(t))
    return out

def first_enter_exit_times(state, times):
    enter_t = None; exit_t = None
    for i in range(1,len(state)):
        if state[i-1]==0 and state[i]==1 and enter_t is None:
            enter_t = (times[i-1]+times[i])/2.0
        if state[i-1]==1 and state[i]==0 and enter_t is not None:
            exit_t = (times[i-1]+times[i])/2.0
            break
    return enter_t, exit_t

def train_and_validate(files=None):  # Accept an externally supplied file list.
    # Use the supplied subset; otherwise fall back to whitelist scanning.
    if files is None:
        files, _ = list_h5_by_group_and_subject_root(DATA_DIR, SELECTED_SUBJECTS)
    if not files:
        raise RuntimeError("未收到可用于训练/验证的 .h5 文件（请检查白名单或路径）。")

    # Training, validation, and checkpointing note.
    subset_files = select_files_balanced_from_subset(files, TOTAL_FILES_TARGET, MIN_PER_CLASS)
    if not subset_files:
        raise RuntimeError("抽样后为空，请放宽 TOTAL_FILES_TARGET 或 MIN_PER_CLASS。")

    # Training, validation, and checkpointing note.
    ds = EEGWindowDataset(
        subset_files,
        win_sec=WIN_SEC, stride_sec=STRIDE_SEC,
        split='val',                  # Training, validation, and checkpointing note.
        train_stride_factor=1,        # Do not trigger training-only masking logic.
        max_windows_per_file=MAX_WINDOWS_PER_FILE,
        ds_step=DS_STEP
    )

    labels = np.array(ds.labels)

    # Training, validation, and checkpointing note.
    rng = np.random.default_rng(42)
    files_unique, seen, file_label = [], set(), {}
    for i, (fp, _, _) in enumerate(ds.index_list):
        if fp not in seen:
            seen.add(fp)
            files_unique.append(fp)
            file_label[fp] = int(labels[i])  # Training, validation, and checkpointing note.

    pos_files = [fp for fp in files_unique if file_label.get(fp, 0) == 1]
    neg_files = [fp for fp in files_unique if file_label.get(fp, 0) == 0]
    rng.shuffle(pos_files); rng.shuffle(neg_files)

    n_val_pos = max(1, int(round(VAL_SPLIT * len(pos_files)))) if len(pos_files) else 0
    n_val_neg = max(1, int(round(VAL_SPLIT * len(neg_files)))) if len(neg_files) else 0
    n_val_pos = min(n_val_pos, max(0, len(pos_files)-1)) if len(pos_files) > 1 else len(pos_files)
    n_val_neg = min(n_val_neg, max(0, len(neg_files)-1)) if len(neg_files) > 1 else len(neg_files)

    val_files   = set(pos_files[:n_val_pos] + neg_files[:n_val_neg])
    train_files = set(files_unique) - val_files

    debug_check_csv_labels(files_unique, max_show=10)

    # Training, validation, and checkpointing note.
    train_idx = np.array([i for i,(fp,_,_) in enumerate(ds.index_list) if fp in train_files], dtype=int)
    val_idx   = np.array([i for i,(fp,_,_) in enumerate(ds.index_list) if fp in val_files],   dtype=int)

    print(f"[Split-by-file] files total={len(files_unique)} | train={len(train_files)} | val={len(val_files)}")

    from collections import defaultdict
    TRAIN_THIN_FACTOR = 2
    by_fp = defaultdict(list)
    for i in train_idx:
        fp, _, _ = ds.index_list[i]
        by_fp[fp].append(i)
    thin = []
    for fp, idxs in by_fp.items():
        thin.extend(idxs[::TRAIN_THIN_FACTOR])
    train_idx = np.array(thin, dtype=int)

    def bincount_safe(arr):
        arr = arr[arr >= 0]
        return np.bincount(arr, minlength=3) 

    print("Class counts (all):  ", bincount_safe(labels))
    print("Class counts (train):", bincount_safe(labels[train_idx]))
    print("Class counts (val):  ", bincount_safe(labels[val_idx]))

    train_loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        sampler=SubsetSequentialSampler(train_idx),
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA, drop_last=True
    )
    val_loader = DataLoader(
        ds, batch_size=BATCH_SIZE,
        sampler=SubsetSequentialSampler(val_idx),
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA
    )

    # Training, validation, and checkpointing note.
    cls_counts = bincount_safe(labels[train_idx])
    w = torch.tensor(1.0 / np.clip(cls_counts, 1, None), dtype=torch.float32, device=DEVICE)  # Training, validation, and checkpointing note.
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    # crit = FocalLoss(alpha=w, gamma=2.0, reduction="mean")

    # Training, validation, and checkpointing note.
    x0, _ = ds[0]
    in_ch = int(x0.shape[0])  # Training, validation, and checkpointing note.
    print(f"[INFO] inferred in_ch = {in_ch}")

    model = EEG_CNN_LSTM(in_ch, N_CLASSES).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)   # Training, validation, and checkpointing note.
    # sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2, factor=0.5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
    optim, max_lr=6e-4, epochs=EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.2, anneal_strategy='cos', div_factor=5.0, final_div_factor=10.0
)
    scaler = GradScaler(enabled=USE_CUDA)

    best_auc = -1.0; patience = 0; best_state = None

    for ep in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; seen_samples = 0
        for b, (x, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=USE_CUDA)
            y = y.to(DEVICE, non_blocking=USE_CUDA)
            optim.zero_grad()
            with autocast(enabled=USE_CUDA):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Training, validation, and checkpointing note.
            scaler.step(optim); scaler.update()
            total_loss += loss.item() * x.size(0)
            seen_samples += x.size(0)
            sched.step()

        print(f"Epoch {ep}/{EPOCHS} | Train Loss: {total_loss/max(1, seen_samples):.4f}")

        model.eval();
        preds, trues_bin = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[:2]
                x = x.to(DEVICE, non_blocking=USE_CUDA)
                logits = model(x)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()  # [B, N_CLASSES]
                p_pos = prob[:, AUC_POS_CLASS] if prob.shape[1] > AUC_POS_CLASS else prob[:, -1]  # Inspect the norm of one representative weight tensor.
                preds.extend(p_pos)
                y_np = y.numpy()
                trues_bin.extend((y_np == AUC_POS_CLASS).astype(int))  

        auc_score = safe_roc_auc(trues_bin, preds)
        scores = np.asarray(preds)

        # Training, validation, and checkpointing note.
        assert len(preds) == len(val_idx), f"len(preds)={len(preds)} vs len(val_idx)={len(val_idx)} 不一致"

        # Training, validation, and checkpointing note.
        with torch.no_grad():
            # Preview whether file-level means vary across files.
            any_w = next(model.parameters())
            wnorm = any_w.detach().float().norm().item()
        print(f"[Diag] weight_norm={wnorm:.6f}")

        # Aggregate predictions by file.
        from collections import defaultdict
        file_pred_count = defaultdict(int)
        for k, i in enumerate(val_idx):
            fp = ds.index_list[i][0]
            file_pred_count[fp] += 1
        print(f"[Diag] val files={len(file_pred_count)} | total val windows by group={sum(file_pred_count.values())}")

        # Training, validation, and checkpointing note.
        file_means_preview = []
        scores_np = np.asarray(preds)
        for k, i in list(enumerate(val_idx))[:min(10, len(val_idx))]:
            fp = ds.index_list[i][0]
            file_means_preview.append(fp)
            if len(file_means_preview) >= 3:
                break
        # Training, validation, and checkpointing note.
        grp = defaultdict(list);
        lab = {}
        for k, i in enumerate(val_idx):
            fp = ds.index_list[i][0]
            grp[fp].append(scores_np[k]);
            lab[fp] = int(labels[i] == AUC_POS_CLASS)
        for fp in list(grp.keys())[:3]:
            print(f"[Diag] {os.path.basename(fp)} mean={np.mean(grp[fp]):.4f} (n={len(grp[fp])})")

        # Training, validation, and checkpointing note.
        pos_files = sum(int(lab[fp] == 1) for fp in grp.keys())
        neg_files = sum(int(lab[fp] == 0) for fp in grp.keys())
        pairs = pos_files * neg_files
        print(f"[Diag] file-level: pos_files={pos_files}, neg_files={neg_files}, pairs={pairs}")

        y_true_file, y_score_file = [], []
        grp = defaultdict(list);
        lab = {}
        for k, i in enumerate(val_idx):
            fp = ds.index_list[i][0]
            grp[fp].append(scores[k])
            lab[fp] = int(labels[i] == AUC_POS_CLASS)
        for fp, sc in grp.items():
            y_true_file.append(int(lab[fp]))
            y_score_file.append(float(np.mean(sc)))
        file_auc = safe_roc_auc(y_true_file, y_score_file)
        print(f"[Val] File-level AUC: {file_auc:.4f}")

        print(f"Val AUC: {auc_score:.4f}")

        sched.step(auc_score if np.isfinite(auc_score) else 0.0)

        if np.isfinite(auc_score) and auc_score > best_auc:
            best_auc = float(auc_score); patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 8:
            print(f"[EarlyStop] no AUC improvement for {patience} epochs; stop.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_to_save, "cnn_eeg_improved_subset.pth")
    print("Model saved → cnn_eeg_improved_subset.pth")

    return model, files

def load_saved_model(model_path, in_ch=8, n_classes=3):
    """
    加载已保存的模型
    """
    # Training, validation, and checkpointing note.
    model = EEG_CNN_LSTM(in_ch, n_classes).to(DEVICE)
    
    # Training, validation, and checkpointing note.
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Training, validation, and checkpointing note.
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        # Remove the module. prefix.
        if all(key.startswith('module.') for key in checkpoint.keys()):
            # Training, validation, and checkpointing note.
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    
    model.eval()  # Training, validation, and checkpointing note.
    print(f"成功加载模型: {model_path}")
    
    return model

# Welch's t-test helpers for significance filtering.
def _welch_t_stat(a: np.ndarray, b: np.ndarray, eps=1e-12):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, np.nan
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1) + eps, np.var(b, ddof=1) + eps
    t = (ma - mb) / np.sqrt(va/na + vb/nb)  # Welch t
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))  # Welch-Satterthwaite degrees of freedom.
    return t, df

def _p_value_from_t(t: float, df: float):
    try:
        from scipy.stats import t as student_t
        return 2.0 * float(student_t.sf(abs(float(t)), df))
    except Exception:
        # Statistical filtering and CPD reporting note.
        from math import erf
        z = abs(float(t))
        p_one = 0.5 * (1.0 - erf(z / np.sqrt(2.0)))
        return 2.0 * p_one

def keep_if_significant_ttest(cp_list, series, times,
                              pre=5.0, post=5.0,
                              alpha=0.05,    # Statistical filtering and CPD reporting note.
                              min_pts=5,     # Statistical filtering and CPD reporting note.
                              min_effect=0.0 # Statistical filtering and CPD reporting note.
                              ):
    series = np.asarray(series, dtype=np.float64)
    kept = []
    for t in cp_list:
        L = [i for i, ti in enumerate(times) if (t - pre) <= ti < t]
        R = [i for i, ti in enumerate(times) if t < ti <= (t + post)]
        if len(L) < min_pts or len(R) < min_pts:
            continue
        a = series[L]; b = series[R]
        if min_effect > 0.0 and abs(np.mean(a) - np.mean(b)) < min_effect:
            continue
        t_stat, df = _welch_t_stat(a, b)
        if not np.isfinite(t_stat) or not np.isfinite(df) or df <= 1:
            continue
        p = _p_value_from_t(t_stat, df)
        if p < alpha:
            kept.append(t)
    return kept



def cpd_on_files(model, files, max_files=None):
    used = files if max_files is None else files[:max_files]
    print("\n=== CPD Summary ===")


    # Whether to save a full change-point timestamp list.
    CP_LIST_MAX     = 10          # True overwrites the output file; False appends.
    CP_DECIMALS     = 2           # Statistical filtering and CPD reporting note.
    SAVE_ALL_CP_TXT = True        # Statistical filtering and CPD reporting note.
    CP_TXT_PATH     = "cp_times_all.txt"
    OVERWRITE_TXT   = True        # Write the header once if requested.

    header = ["filename", "mean_p", "std_p", "ratio1_prob", "n_cp_final", "cp_times_s", "note"]
    print("{:<28} {:>7} {:>7} {:>12} {:>11}  {:<40} {}".format(*header))


    # Statistical filtering and CPD reporting note.
    if SAVE_ALL_CP_TXT and OVERWRITE_TXT:
        with open(CP_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("# file\tchangepoint_times_seconds (final after merge+filter)\n")

    # Statistical filtering and CPD reporting note.
    def merge_cps(a, b, min_sep=1.0):
        xs = sorted(list(a) + list(b))
        out = []
        for t in xs:
            if not out or t - out[-1] >= min_sep:
                out.append(t)
        return out

    def cosine_distance(a, b, eps=1e-8):
        a = np.asarray(a);
        b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + eps)
        return 1.0 - float(np.dot(a, b) / denom)


    def moving_average(x, k):
        if k <= 1: return np.asarray(x)
        k = int(k)
        w = np.ones(k, dtype=np.float32) / k
        return np.convolve(np.asarray(x, dtype=np.float32), w, mode='same')

    for fp in used:
        # Statistical filtering and CPD reporting note.
        times, probs, Fs = predict_probs_on_file(fp, model, WIN_SEC, STRIDE_SEC, DS_STEP, DEVICE)
        if len(times) == 0:
            print("{:<28} {:>7} {:>7} {:>12} {:>11}  {:<40} {}".format(
                Path(fp).name[:28], "NA", "NA", "NA", 0, "too_short", ""
            ))
            if SAVE_ALL_CP_TXT:
                with open(CP_TXT_PATH, "a", encoding="utf-8") as f:
                    f.write(f"{Path(fp).name}\tTOO_SHORT\n")
            continue

        # Statistical filtering and CPD reporting note.
        k_prob = max(1, int(round(SMOOTH_SEC / max(STRIDE_SEC, 1e-6))))
        p_smooth = moving_average(probs, k_prob)

        # Statistical filtering and CPD reporting note.
        t_feat, feats, _ = predict_feats_on_file(fp, model, WIN_SEC, STRIDE_SEC, DS_STEP, DEVICE)
        if feats is None or len(feats) < 3:
            cps_final = []
        else:
            if len(t_feat) != len(times) or not np.allclose(t_feat, times):
                feats_aligned = np.zeros((len(times), feats.shape[1]), dtype=np.float32)
                t_left, t_right = t_feat[0], t_feat[-1]
                mask = (times >= t_left) & (times <= t_right)
                for d in range(feats.shape[1]):
                    feats_aligned[mask, d] = np.interp(times[mask], t_feat, feats[:, d])
                feats_aligned[times < t_left]  = feats_aligned[np.where(mask)[0][0]]
                feats_aligned[times > t_right] = feats_aligned[np.where(mask)[0][-1]]
                X = feats_aligned
            else:
                X = feats
            X = X.astype(np.float32)
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
            if DIST_SMOOTH_SEC > 0:
                k_dist = max(1, int(round(DIST_SMOOTH_SEC / max(STRIDE_SEC, 1e-6))))
                if k_dist > 1 and X.shape[0] >= k_dist:
                    X = np.vstack([moving_average(X[:, i], k_dist) for i in range(X.shape[1])]).T

            
            min_size = max(2, int(round(RUPTURES_MIN_STATE_SEC / max(STRIDE_SEC, 1e-6))))
            try:
                import ruptures as rpt
                if RUPTURES_BACKEND.lower() == "pelt":
                    algo = rpt.Pelt(model=RUPTURES_MODEL, min_size=min_size, jump=RUPTURES_JUMP).fit(X)
                    bkps = algo.predict(pen=float(RUPTURES_PEN))
                else:
                    algo = rpt.Binseg(model=RUPTURES_MODEL, min_size=min_size, jump=RUPTURES_JUMP).fit(X)
                    bkps = algo.predict(n_bkps=int(RUPTURES_N_BKPS))
            except Exception as e:
                print(f"[ruptures] failed: {e}")
                bkps = [len(X)] 
            cps_rpt = [ times[i-1] for i in bkps if 1 <= i < len(times) ]

            Xc = X - X.mean(0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(Xc, full_matrices=False)
                pc1 = Xc @ vh[0]  # [T]
                series_1d = pc1
            except Exception:
                series_1d = Xc[:, 0]     
            cps_lock = _apply_refractory(sorted(cps_rpt), refract_sec=4.0)
            cps_lock = remove_cps_near_bounds(cps_lock, bounds=None, pad_sec=3.0)
            cps_final = keep_if_significant_ttest(
                cps_lock, series_1d, times, pre=3.0, post=3.0, alpha=0.05, min_pts=3, min_effect=0.0
            )

        # Statistical filtering and CPD reporting note.
        cp_print = ", ".join([f"{t:.{CP_DECIMALS}f}" for t in cps_final[:CP_LIST_MAX]])
        cp_print += (" ..." if len(cps_final) > CP_LIST_MAX else "")
        row = [
            Path(fp).name[:28],
            f"{np.mean(p_smooth):.3f}",
            f"{np.std(p_smooth):.3f}",
            f"{(np.mean(p_smooth > 0.5)) * 100:5.1f}%",
            f"{len(cps_final):3d}",
            cp_print,
            ""
        ]
        print("{:<28} {:>7} {:>7} {:>12} {:>11}  {:<40} {}".format(*row))

        if SAVE_ALL_CP_TXT:
            with open(CP_TXT_PATH, "a", encoding="utf-8") as f:
                f.write(Path(fp).name + "\t" + ",".join([f"{t:.3f}" for t in cps_final]) + "\n")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 5))
        plt.plot(times, p_smooth, label=f"p(smooth,{SMOOTH_SEC:.1f}s)")
        first = True
        for t in cps_final:
            plt.axvline(t, color="red", linewidth=1.2, label="CP(final)" if first else None); first=False
        plt.xlabel("Time (s)"); plt.ylabel("Prob(class=1)")
        plt.title(f"CPD (feature-space · ruptures) — {Path(fp).name}")
        plt.legend(loc="lower right"); plt.tight_layout(); plt.show()

def load_concat_sig(files, ch_slice=CHANNEL_SLICE):
    sig_list = []
    Fs = None
    for fp in files:
        with h5py.File(fp, "r") as f:
            arr = np.array(f["/sig"][ch_slice, :])
            fs  = safe_item(f["/sig"].attrs["Fs"])
        if Fs is None:
            Fs = fs
        sig_list.append(arr)
    sig = np.concatenate(sig_list, axis=1)
    return sig, Fs

def time_slot_from_name(fp: str) -> str:
    """从文件名解析出可读的时间档位标签（尽量覆盖你的命名习惯）"""
    n = _norm_lower(fp)
    # Visualization or main-pipeline note.
    for k in ["30min", "20min", "10min"]:
        if f"iv级前{k}" in n or f"iv級前{k}" in n:
            return f"{k} before SE"
    # Visualization or main-pipeline note.
    if "止惊前10min" in n or "止驚前10min" in n: return "10min before DZP"
    if "止惊后10min" in n or "止驚後10min" in n: return "10min after DZP"
    for k in ["1h","2h","3h"]:
        if f"止惊后{k}" in n or f"止驚後{k}" in n:
            return f"{k} after DZP"
    # Visualization or main-pipeline note.
    for k in ["1d","3d","7d","28d"]:
        if f"建模后{k}" in n or f"建模後{k}" in n:
            return f"{k} after SE"
    # Pre-modeling time slot.
    if "建模前" in n or "baseline" in n: return "baseline"
    return os.path.basename(fp)

def _pe_over_windows(fp: str, ch_idx: int, win_sec: float = 1.0, hop_sec: float = 1.0, m: int = 3, tau: int = 1):
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])
        Fs = safe_item(f["/sig"].attrs["Fs"])
    x = sig[ch_idx]  # Visualization or main-pipeline note.
    win = int(round(win_sec * Fs)); hop = int(round(hop_sec * Fs))
    vals = []
    for s in range(0, max(1, x.size - win + 1), max(1, hop)):
        seg = x[s:s + win]
        if seg.size < win: break
        vals.append(permutation_entropy(seg, m=m, tau=tau))
    return np.asarray(vals, dtype=np.float32)

def boxplot_pe_by_timeslots(subject: str, ch_idx: int = 0, win_sec: float = 1.0, hop_sec: float = 1.0,
                            m: int = 3, tau: int = 1, save_path: str = None):
    all_files, _ = list_h5_by_group_and_subject_root(DATA_DIR, SELECTED_SUBJECTS)
    groups = group_files_by_subject_then_state(all_files, labels_csv=CSV_PATH)
    sd = groups.get(subject, None)
    if not sd:
        print(f"[PE] 找不到个体 {subject}"); return

    # Visualization or main-pipeline note.
    seq = (sd.get("状态1_建模前", []) + sd.get("状态2_IV级前", []) +
           sd.get("状态3_止惊前后", []) + sd.get("状态4_建模后", []))
    if not seq:
        print(f"[PE] {subject} 无文件"); return

    slots, data = [], []
    for fp in seq:
        slot = time_slot_from_name(fp)
        vals = _pe_over_windows(fp, ch_idx, win_sec, hop_sec, m, tau)
        if vals.size == 0: 
            continue
        slots.append(slot); data.append(vals)

    if not data:
        print(f"[PE] 没有可画的数据"); return

    # Visualization or main-pipeline note.
    order = [
        "baseline",
        "30min before SE", "20min before SE", "10min before SE",
        "10min before DZP", "10min after DZP",
        "1h after DZP", "2h after DZP", "3h after DZP",
        "1d after SE", "3d after SE", "7d after SE", "28d after SE"
    ]
    # Visualization or main-pipeline note.
    idx = sorted(range(len(slots)), key=lambda i: (order.index(slots[i]) if slots[i] in order else 999, i))
    slots_sorted = [slots[i] for i in idx]
    data_sorted  = [data[i]  for i in idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data_sorted, showfliers=True, whis=1.5)
    ax.set_xticks(range(1, len(slots_sorted)+1))
    ax.set_xticklabels(slots_sorted, rotation=35, ha="right")
    ax.set_ylabel("Permutation Entropy (normalized)")
    ax.set_title(f"{subject} - Channel {ch_idx} | window={win_sec:.1f}s hop={hop_sec:.1f}s, m={m}, tau={tau}")

    # Visualization or main-pipeline note.
    # Visualization or main-pipeline note.
    def _try_vline(label):
        if label in slots_sorted:
            i = slots_sorted.index(label) + 0.5
            ax.axvline(i, ls="--", lw=1.0, alpha=0.5, color="k")
    for label in ["10min before SE", "10min after DZP", "3h after DZP", "3d after SE", "7d after SE"]:
        _try_vline(label)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"[Saved] {save_path}")
    plt.show()

CH_NAMES_8 = ["P4","P3","C4","C3","F4","F3","Fp2","Fp1"]

from scipy.signal import welch
def bandpower_welch(sig, Fs, fmin=200, fmax=500):
    """sig: [C, N]"""
    nperseg = min(int(Fs*2), sig.shape[1])  # Two-second Welch window.
    psd_list = []
    for c in range(sig.shape[0]):
        f, Pxx = welch(sig[c], fs=Fs, nperseg=nperseg)
        mask = (f >= fmin) & (f <= fmax)
        psd_list.append(Pxx[mask].mean())
    return np.array(psd_list)

TOPO_POS = {
    "P4": (-0.5,  1.0),
    "P3": ( 0.5,  1.0),
    "C4": (-0.5,  0.3),
    "C3": ( 0.5,  0.3),
    "F4": (-0.5, -0.4),
    "F3": ( 0.5, -0.4),
    "Fp2": (-0.5, -1.0),
    "Fp1": ( 0.5, -1.0),
}

def plot_hfo_topography_for_subject(group, subj, files,
                                    fmin=200, fmax=500):
    sig, Fs = load_concat_sig(files, ch_slice=CHANNEL_SLICE)
    power = bandpower_welch(sig, Fs, fmin=fmin, fmax=fmax)

    # Visualization or main-pipeline note.
    power = power[:len(CH_NAMES_8)]

    xs = [TOPO_POS[ch][0] for ch in CH_NAMES_8]
    ys = [TOPO_POS[ch][1] for ch in CH_NAMES_8]

    plt.figure(figsize=(6,6))
    sc = plt.scatter(xs, ys, c=power, s=500, cmap="jet")
    for x, y, name in zip(xs, ys, CH_NAMES_8):
        plt.text(x, y, name, color="white", ha="center", va="center", fontsize=10, weight="bold")
    plt.colorbar(sc, label="200–500 Hz Power")
    plt.title(f"{subj} {group} – HFO Power Topography (200–500 Hz)")
    plt.axis("off")
    plt.show()

    out = os.path.join(PLOT_DIR, f"topo_{_safe_name(group)}_{_safe_name(subj)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("[Topo] saved:", out)

def plot_group_cumulative_cps(group_cp_store, group_name, block_name,
                              t_step=10.0):
    key = (group_name, block_name)
    entries = group_cp_store.get(key, [])
    if not entries:
        print("[GroupCum] no entries for", key)
        return

    # Visualization or main-pipeline note.
    T_min = min(e["dur"] for e in entries)
    grid = np.arange(0.0, T_min + 1e-6, t_step)  # 0,10,20,...

    curves = []
    for e in entries:
        cps = np.array(sorted(e["cps"]))
        # Visualization or main-pipeline note.
        counts = [np.sum(cps <= t) for t in grid]
        curves.append(np.array(counts, dtype=float))
    curves = np.vstack(curves)       # [n_subj, n_time]
    mean_curve = curves.mean(axis=0)

    plt.figure(figsize=(6,4))
    for c in curves:
        plt.plot(grid/60.0, c, alpha=0.3)   # Visualization or main-pipeline note.
    plt.plot(grid/60.0, mean_curve, linewidth=2, label="Group mean")
    plt.xlabel("Time since block start (min)")
    plt.ylabel("Cumulative number of change points")
    plt.title(f"{group_name} – Block {block_name} cumulative CPs")
    plt.legend()
    plt.show()

    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f"cumcp_{_safe_name(group_name)}_block{block_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("[GroupCum] saved:", out)


# Visualization or main-pipeline note.
def time_slot_from_name(fp: str) -> str:
    """从文件名解析出可读的时间档位标签（尽量覆盖你的命名习惯）"""
    n = _norm_lower(fp)
    # Visualization or main-pipeline note.
    for k in ["30min", "20min", "10min"]:
        if f"iv级前{k}" in n or f"iv級前{k}" in n:
            return f"{k} before SE"
    # Visualization or main-pipeline note.
    if "止惊前10min" in n or "止驚前10min" in n: return "10min before DZP"
    if "止惊后10min" in n or "止驚後10min" in n: return "10min after DZP"
    for k in ["1h","2h","3h"]:
        if f"止惊后{k}" in n or f"止驚後{k}" in n:
            return f"{k} after DZP"
    # Visualization or main-pipeline note.
    for k in ["1d","3d","7d","28d"]:
        if f"建模后{k}" in n or f"建模後{k}" in n:
            return f"{k} after SE"
    # Visualization or main-pipeline note.
    if "建模前" in n or "baseline" in n: return "baseline"
    return os.path.basename(fp)

def _pe_over_windows(fp: str, ch_idx: int, win_sec: float = 1.0, hop_sec: float = 1.0, m: int = 3, tau: int = 1):
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])
        Fs = safe_item(f["/sig"].attrs["Fs"])
    x = sig[ch_idx]  # Visualization or main-pipeline note.
    win = int(round(win_sec * Fs)); hop = int(round(hop_sec * Fs))
    vals = []
    for s in range(0, max(1, x.size - win + 1), max(1, hop)):
        seg = x[s:s + win]
        if seg.size < win: break
        vals.append(permutation_entropy(seg, m=m, tau=tau))
    return np.asarray(vals, dtype=np.float32)

def build_model_for_inference(checkpoint_path):
    in_ch = 8              
    n_classes = N_CLASSES  

    model = EEG_CNN_LSTM(in_ch, n_classes).to(DEVICE)

    # Visualization or main-pipeline note.
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()

    # Visualization or main-pipeline note.
    predict_probs_on_signal._model = model
    predict_probs_on_signal._device = DEVICE

    return model

def extract_embeddings_from_files(model, file_list, win_sec, stride_sec, ds_step, device):
    all_Z = []
    all_t = []
    all_fileidx = []
    for i, fp in enumerate(file_list):
        # Visualization or main-pipeline note.
        # Assume predict_feats_on_file returns times, features, and Fs.
        times, feats, Fs = predict_feats_on_file(fp, model, win_sec, stride_sec, ds_step, device)
        if feats is None or len(feats) == 0:
            continue
        all_Z.append(feats)
        all_t.append(times)
        all_fileidx.append(np.full(len(times), i, dtype=int))
    if not all_Z:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,), dtype=int)
    Z = np.vstack(all_Z)
    times = np.concatenate(all_t)
    file_idx = np.concatenate(all_fileidx)
    return Z, times, file_idx

def compute_silhouette(Z, labels):
    """
    计算嵌入向量的轮廓系数
    """
    try:
        from sklearn.metrics import silhouette_score
        # Visualization or main-pipeline note.
        if len(np.unique(labels)) < 2 or len(Z) < 3:
            return float('nan')
        return float(silhouette_score(Z, labels))
    except Exception as e:
        print(f"[silhouette] failed: {e}")
        return float('nan')

from sklearn.manifold import TSNE
def plot_embedding_2d(Z, labels, methods=["pca", "tsne"], title_prefix="Embeddings", save_dir=None):
    """
    不使用UMAP的2D嵌入可视化
    """
    if Z.size == 0:
        print("[plot_embedding_2d] empty embeddings")
        return
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            Z2 = reducer.fit_transform(Z)
            variance_ratio = reducer.explained_variance_ratio_
            title = f"{title_prefix} PCA\n(Var: {variance_ratio[0]:.1%}+{variance_ratio[1]:.1%})"
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, 
                          init="pca", random_state=42)
            Z2 = reducer.fit_transform(Z)
            title = f"{title_prefix} t-SNE"
        else:
            continue
            
        ax = axes[i]
        for c in np.unique(labels):
            idx = labels == c
            ax.scatter(Z2[idx,0], Z2[idx,1], s=12, alpha=0.7, label=f'File {c}')
        
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()

    fname = f"{title_prefix.replace(' ', '_')}_comparison.png"
    save_path = os.path.join(save_dir, fname)
    plt.show()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {save_path}")

    silhouette_pca = compute_silhouette(Z, labels)
    print(f"Silhouette Score (by file): {silhouette_pca:.3f}")
    
    # if save_dir:
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_comparison.png")
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     print(f"[Saved] {save_path}")
    
    # plt.show()
    
    # Visualization or main-pipeline note.
    # silhouette_pca = compute_silhouette(Z, labels)
    # print(f"Silhouette Score (by file): {silhouette_pca:.3f}")

def enhanced_tsne_visualization(Z, labels, file_names=None, perplexities=[15, 30, 50]):
    """
    使用不同perplexity参数的t-SNE对比可视化
    """
    n_perplexities = len(perplexities)
    fig, axes = plt.subplots(1, n_perplexities, figsize=(5*n_perplexities, 5))
    if n_perplexities == 1:
        axes = [axes]
    
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   max_iter=1500, random_state=42, init='pca')
        Z_tsne = tsne.fit_transform(Z)
        
        ax = axes[i]
        scatter = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=labels, 
                           cmap='tab10', s=20, alpha=0.7)
        ax.set_title(f't-SNE (perplexity={perplexity})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return Z_tsne

def temporal_embedding_visualization(Z, times, file_idx, 
                                     n_files_to_show=5,
                                     cps=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Use a common mask for the first 600 seconds.
    times = np.asarray(times)
    file_idx = np.asarray(file_idx)
    mask_600 = times <= 600.0      # Keep only windows in the 0-600 s interval.
    
    # Temporal PCA projection using PC1.
    pca = PCA(n_components=1, random_state=42)
    Z_pca1d = pca.fit_transform(Z)   # [N, 1]
    
    unique_files = np.unique(file_idx)[:n_files_to_show]
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_files)))
    
    for i, file_id in enumerate(unique_files):
        # Visualization or main-pipeline note.
        mask = (file_idx == file_id) & mask_600
        if not np.any(mask):
            continue
        ax1.scatter(times[mask], Z_pca1d[mask],
                    c=[colors[i]], label=f'File {file_id}', s=10, alpha=0.6)
    
    ax1.set_xlim(0, 600)   # Visualization or main-pipeline note.
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('PC1 Value')
    ax1.set_title('Temporal Evolution of Principal Component 1 (first 600 s)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Visualization or main-pipeline note.
    pca2d = PCA(n_components=2, random_state=42)
    Z_pca2d = pca2d.fit_transform(Z)
    
    for i, file_id in enumerate(unique_files):
        mask = (file_idx == file_id) & mask_600
        if not np.any(mask):
            continue
        ax2.scatter(Z_pca2d[mask, 0], Z_pca2d[mask, 1],
                    c=[colors[i]], label=f'File {file_id}', s=15, alpha=0.6)
    
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('2D PCA Colored by File (first 600 s)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    
    # Visualization or main-pipeline note.
    variance_ratio = pca2d.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1={variance_ratio[0]:.1%}, PC2={variance_ratio[1]:.1%}")
    
    return Z_pca2d

from itertools import combinations
from scipy.signal import coherence

def compute_coherence_matrix(sig, Fs, fmin=200, fmax=500):
    C = sig.shape[0]
    coh_mat = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(i, C):
            if i == j:
                coh_mat[i, j] = 1.0
            else:
                f, Cxy = coherence(sig[i], sig[j], fs=Fs, nperseg=min(int(Fs*2), sig.shape[1]))
                mask = (f >= fmin) & (f <= fmax)
                coh_val = Cxy[mask].mean()
                coh_mat[i, j] = coh_mat[j, i] = coh_val
    return coh_mat

def plot_coherence_matrix_for_subject(group, subj, files,
                                      fmin=200, fmax=500):
    sig, Fs = load_concat_sig(files, ch_slice=CHANNEL_SLICE)
    C = min(sig.shape[0], len(CH_NAMES_8))
    sig = sig[:C]
    coh = compute_coherence_matrix(sig, Fs, fmin=fmin, fmax=fmax)

    plt.figure(figsize=(6,5))
    im = plt.imshow(coh, origin="lower", vmin=0, vmax=1, cmap="jet")
    plt.colorbar(im, label="Coherence")
    ticks = range(C)
    labels = CH_NAMES_8[:C]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.title(f"{subj} {group} – 200–500 Hz Coherence")
    out = os.path.join(PLOT_DIR, f"coh_{_safe_name(group)}_{_safe_name(subj)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print("[Coherence] saved:", out)

 
def evaluate_embedding_quality(Z, labels):
    """
    评估嵌入质量的多指标评估
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    
    results = {}
    
    # Visualization or main-pipeline note.
    results['silhouette'] = compute_silhouette(Z, labels)
    
    # Visualization or main-pipeline note.
    try:
        results['calinski_harabasz'] = calinski_harabasz_score(Z, labels)
    except:
        results['calinski_harabasz'] = float('nan')
    
    # Visualization or main-pipeline note.
    try:
        results['davies_bouldin'] = davies_bouldin_score(Z, labels)
    except:
        results['davies_bouldin'] = float('nan')
    
    # Visualization or main-pipeline note.
    unique_labels = np.unique(labels)
    intra_dists = []
    inter_dists = []
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 1:
            # Visualization or main-pipeline note.
            intra_dist = np.mean(np.linalg.norm(Z[mask] - np.mean(Z[mask], axis=0), axis=1))
            intra_dists.append(intra_dist)
            
            # Visualization or main-pipeline note.
            other_mask = labels != label
            if np.sum(other_mask) > 0:
                inter_dist = np.linalg.norm(np.mean(Z[mask], axis=0) - np.mean(Z[other_mask], axis=0))
                inter_dists.append(inter_dist)
    
    results['mean_intra_distance'] = np.mean(intra_dists) if intra_dists else float('nan')
    results['mean_inter_distance'] = np.mean(inter_dists) if inter_dists else float('nan')
    results['separation_ratio'] = results['mean_inter_distance'] / results['mean_intra_distance'] if intra_dists else float('nan')
    
    print("=== Embedding Quality Evaluation ===")
    for metric, value in results.items():
        print(f"{metric:>20}: {value:.4f}")
    
    return results

group_cp_store = {}   # key: (group_name, block_name) -> list of dict(subject=..., cps=..., dur=...)


def time_slot_from_name(fp: str) -> str:
    """从文件名解析出可读的时间档位标签（尽量覆盖你的命名习惯）"""
    n = _norm_lower(fp)
    # Visualization or main-pipeline note.
    for k in ["30min", "20min", "10min"]:
        if f"iv级前{k}" in n or f"iv級前{k}" in n:
            return f"{k} before SE"
    # Visualization or main-pipeline note.
    if "止惊前10min" in n or "止驚前10min" in n: return "10min before DZP"
    if "止惊后10min" in n or "止驚後10min" in n: return "10min after DZP"
    for k in ["1h","2h","3h"]:
        if f"止惊后{k}" in n or f"止驚後{k}" in n:
            return f"{k} after DZP"
    # Visualization or main-pipeline note.
    for k in ["1d","3d","7d","28d"]:
        if f"建模后{k}" in n or f"建模後{k}" in n:
            return f"{k} after SE"
    # Visualization or main-pipeline note.
    if "建模前" in n or "baseline" in n: return "baseline"
    return os.path.basename(fp)

def _pe_over_windows(fp: str, ch_idx: int, win_sec: float = 1.0, hop_sec: float = 1.0, m: int = 3, tau: int = 1):
    with h5py.File(fp, "r") as f:
        sig = np.array(f["/sig"][CHANNEL_SLICE, :])
        Fs = safe_item(f["/sig"].attrs["Fs"])
    x = sig[ch_idx]  # Visualization or main-pipeline note.
    win = int(round(win_sec * Fs)); hop = int(round(hop_sec * Fs))
    vals = []
    for s in range(0, max(1, x.size - win + 1), max(1, hop)):
        seg = x[s:s + win]
        if seg.size < win: break
        vals.append(permutation_entropy(seg, m=m, tau=tau))
    return np.asarray(vals, dtype=np.float32)



def main():
    subset_files, subj_map = list_h5_by_group_and_subject_root(DATA_DIR, SELECTED_SUBJECTS)
    if not subset_files:
        raise RuntimeError("白名单指定的组/个体下未匹配到 .h5 文件。请检查路径命名。")

    print(f"[Info] 将在这些个体的数据上训练与检测，文件总数 = {len(subset_files)}")

    # Use the saved model for visualization instead of retraining.
    model = load_saved_model("cnn_eeg_improved_subset.pth")
    files_for_vis = [fp for fp in subset_files if "47号" in os.path.basename(fp)]
    if not files_for_vis:
        files_for_vis = subset_files[:6]  # Fallback to the first six files if subject 47 is unavailable.

    Z, times, file_idx = extract_embeddings_from_files(
        model, files_for_vis, 
        win_sec=WIN_SEC, stride_sec=STRIDE_SEC, 
        ds_step=DS_STEP, device=DEVICE
    )
    print(f"Embeddings shape: {Z.shape}")
    # run_audit_over_dir(DATA_DIR, model, out_csv="audit_invariance.csv")
    # Visualization or main-pipeline note.
    plot_subject_three_phases("47号", ch_idx=0, sec=1.0, save_path=r"results\three_phases_example.png")
    boxplot_pe_by_timeslots("47号", ch_idx=0, win_sec=1.0, hop_sec=1.0, m=3, tau=1,
                        save_path=r"results\pe_boxplot_example.png")
    plot_embedding_2d(
        Z, file_idx, 
        methods=["pca", "tsne"],
        title_prefix="47号 Embeddings",
        save_dir=r"results"
    )
#     cps_vis, sig_cat_vis, Fs_vis = _run_cpd_on_concat(
#     model, files_for_vis,
# Optional visualization and quality-evaluation calls.
#     prob_png=None,
#     heat_png=None,
#     return_sig=True
# )
#     enhanced_tsne_visualization(Z, file_idx, perplexities=[10, 30, 50])
#     temporal_embedding_visualization(Z, times, file_idx, n_files_to_show=min(5, len(np.unique(file_idx))), cps=cps_vis)
#     quality_metrics = evaluate_embedding_quality(Z, file_idx)


    cpd_on_subject_two_merged_blocks(model, subset_files, save_txt="cp_times_two_blocks.txt")
    # cpd_on_subject_full_timeline(model, subset_files, save_txt="cp_times_full.txt")
if __name__ == "__main__":
    main()

