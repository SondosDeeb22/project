"""Image collection: upload / Hugging Face loading, lazy decode, caching.

Memory rule: the session keeps only metadata. Every image lives as a file in a
per-session temp folder and is decoded on demand (current image + visible
thumbnails only).
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import uuid

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import processing as P

MAX_SIDE = 2048
EXTS = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
GRAY_GIF = "data:image/gif;base64,R0lGODlhAQABAIAAAC8vLwAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="


# ----------------------------------------------------------------- state ----
def init_state():
    defaults = {
        "imgs": [], "cur": 0, "page": 0, "tech": "negative", "results": {}, "job": None,
        "up_gen": 0, "tab": "Workbench", "grid_page": 0, "zip_path": None, "workdir": None,
        "processed_images": {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def page_size() -> int:
    return 8 if len(st.session_state.imgs) > 10 else 10


def workdir() -> str:
    if not st.session_state.workdir or not os.path.isdir(st.session_state.workdir):
        st.session_state.workdir = tempfile.mkdtemp(prefix="imagelab_")
    return st.session_state.workdir


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def find(item_id: str):
    for it in st.session_state.imgs:
        if it["id"] == item_id:
            return it
    return None


def select(i: int):
    n = len(st.session_state.imgs)
    st.session_state.cur = max(0, min(i, n - 1))
    st.session_state.page = st.session_state.cur // page_size()


def _write(name: str, blob: bytes, source: str, label: str | None = None) -> dict:
    item_id = uuid.uuid4().hex[:10]
    path = os.path.join(workdir(), f"{item_id}.img")
    with open(path, "wb") as f:
        f.write(blob)
    it = {"id": item_id, "name": name, "size": len(blob), "path": path, "source": source, "label": label}
    st.session_state.imgs.append(it)
    return it


def add_files(files) -> int:
    """Append uploaded files (never replaces). Skips duplicates by name + size."""
    s = st.session_state
    seen = {(i["name"], i["size"]) for i in s.imgs if i["source"] == "upload"}
    added = 0
    for f in files:
        blob = f.getvalue()
        if (f.name, len(blob)) in seen:
            continue
        _write(f.name, blob, "upload")
        seen.add((f.name, len(blob)))
        added += 1
    # Clear processed results when new files are added
    if added > 0:
        s.results = {}
        s.processed_images = {}
        s.zip_path = None
    return added


def remove(item_id: str):
    s = st.session_state
    idx = next((i for i, it in enumerate(s.imgs) if it["id"] == item_id), None)
    if idx is None:
        return
    it = s.imgs.pop(idx)
    try:
        os.remove(it["path"])
    except OSError:
        pass
    s.results.pop(item_id, None)
    s.processed_images.pop(item_id, None)
    if idx < s.cur:
        s.cur -= 1
    s.cur = max(0, min(s.cur, len(s.imgs) - 1))
    s.page = s.cur // page_size()


def load_hf(ds_id: str, split: str, limit: int, stream: bool, progress=None) -> int:
    """Stream (or load) up to `limit` images from a Hugging Face dataset.
    Raw bytes go straight to disk, so memory stays flat."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(ds_id, split=split, streaming=stream)
    feats = ds.features
    col = None
    if feats:
        col = next((k for k, v in feats.items() if isinstance(v, HFImage)), None)
    label_names = None
    if feats and "label" in feats and hasattr(feats["label"], "names"):
        label_names = feats["label"].names
    if col is not None:
        ds = ds.cast_column(col, HFImage(decode=False))
    it = ds.take(limit) if stream else ds.select(range(min(limit, len(ds))))
    total = limit if stream else min(limit, len(ds))
    base = ds_id.split("/")[-1]
    added = 0
    for i, ex in enumerate(it):
        try:
            if col is not None:
                d = ex[col]
                blob = d.get("bytes")
                if blob is None and d.get("path"):
                    with open(d["path"], "rb") as fh:
                        blob = fh.read()
            else:  # features unknown: look for a PIL image in the example
                pil = next((v for v in ex.values() if isinstance(v, Image.Image)), None)
                if pil is None:
                    raise ValueError("No image column found in this dataset.")
                buf = io.BytesIO()
                pil.save(buf, "PNG")
                blob = buf.getvalue()
            lab = ex.get("label")
            if label_names and isinstance(lab, int) and 0 <= lab < len(label_names):
                lab = label_names[lab]
            _write(f"{base}_{i:05d}", blob, "hf", None if lab is None else str(lab))
            added += 1
        except Exception as e:  # skip bad rows, keep going
            if col is None and "No image column" in str(e):
                raise
        if progress and (i % 10 == 0 or i + 1 == total):
            progress(i + 1, total)
    if added == 0:
        raise ValueError("No images could be read from this dataset/split.")
    # Clear processed results when new images are loaded
    if added > 0:
        st.session_state.results = {}
        st.session_state.processed_images = {}
        st.session_state.zip_path = None
    return added


# --------------------------------------------------------------- decoding ---
def load_gray(path: str, max_side: int | None = MAX_SIDE):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        size = im.size
        if im.mode in ("I;16", "I", "F"):
            a = np.asarray(im, dtype=np.float64)
            rng = a.max() - a.min()
            a = (a - a.min()) / (rng if rng else 1.0) * 255.0
            g = Image.fromarray(a.astype(np.uint8), "L")
        else:
            g = im.convert("L")
        if max_side and max(g.size) > max_side:
            g.thumbnail((max_side, max_side))
        return np.asarray(g, dtype=np.uint8).copy(), size


@st.cache_data(show_spinner=False, max_entries=16)
def load_gray_cached(path: str):
    return load_gray(path, MAX_SIDE)


def to_uri(arr: np.ndarray, max_side: int = 1100, quality: int = 92) -> str:
    im = Image.fromarray(arr, "L")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _small_gray(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        try:
            im.draft("L", (size * 2, size * 2))
        except Exception:
            pass
        g = ImageOps.exif_transpose(im).convert("L")
        g = ImageOps.fit(g, (size, size))
        return np.asarray(g, dtype=np.uint8)


@st.cache_data(show_spinner=False, max_entries=3000)
def thumb_uri(path: str, size: int = 64) -> str:
    try:
        return to_uri(_small_gray(path, size), size, 80)
    except Exception:
        return GRAY_GIF


@st.cache_data(show_spinner=False, max_entries=600)
def tile_uri(path: str, tech: str, params: tuple, size: int = 160) -> str:
    try:
        arr = _small_gray(path, size)
        return to_uri(P.get(tech).func(arr, **dict(params)), size, 85)
    except Exception:
        return GRAY_GIF


@st.cache_data(show_spinner=False, max_entries=400)
def card_uris(path: str, tech: str, params: tuple, size: int = 360):
    """(original, result) data-URIs for a grid card. Aspect ratio is preserved."""
    try:
        with Image.open(path) as im:
            try:
                im.draft("L", (size * 2, size * 2))
            except Exception:
                pass
            g = ImageOps.exif_transpose(im).convert("L")
        g.thumbnail((size, size))
        a = np.asarray(g, dtype=np.uint8).copy()
        out = P.get(tech).func(a, **dict(params))
        return to_uri(a, size, 85), to_uri(out, size, 85)
    except Exception:
        return GRAY_GIF, GRAY_GIF
