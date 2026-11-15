"""Chunked batch runner (driven by a fragment) + exports. No output images are kept."""
from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import streamlit as st
from PIL import Image

import data
import processing as P


def params_for(t: P.Transform) -> dict:
    return {v.key: st.session_state.get(f"v_{t.key}_{v.key}", v.default) for v in t.vars}


def start_job():
    s = st.session_state
    t = P.get(s.tech)
    s.results = {}
    s.zip_path = None
    s.processed_images = {}  # Store transformed images
    s.job = {"active": True, "ids": [i["id"] for i in s.imgs], "pos": 0, "tech": t.key,
             "params": params_for(t), "t0": time.time(), "done": 0, "failed": 0}


def cancel_job():
    if st.session_state.job:
        st.session_state.job["active"] = False
    st.session_state.processed_images = {}


def step(budget: float = 0.45) -> bool:
    """Process items for ~budget seconds. Returns True when the job is finished."""
    s = st.session_state
    job = s.job
    if not job or not job["active"]:
        return True
    t = P.get(job["tech"])
    end = time.perf_counter() + budget
    while job["pos"] < len(job["ids"]) and time.perf_counter() < end:
        item = data.find(job["ids"][job["pos"]])
        job["pos"] += 1
        if item is None:
            continue
        try:
            arr, _ = data.load_gray(item["path"], None)
            t0 = time.perf_counter()
            transformed = t.func(arr, **job["params"])
            ms = (time.perf_counter() - t0) * 1000
            s.results[item["id"]] = {"status": "done", "ms": round(ms, 2), "tech": t.name}
            # Store transformed image for zip download
            s.processed_images[item["id"]] = transformed
            job["done"] += 1
        except Exception as e:
            s.results[item["id"]] = {"status": "failed", "ms": None, "tech": t.name, "error": str(e)[:80]}
            job["failed"] += 1
    if job["pos"] >= len(job["ids"]):
        job["active"] = False
        return True
    return False


def table() -> pd.DataFrame:
    rows = []
    for i, it in enumerate(st.session_state.imgs):
        r = st.session_state.results.get(it["id"])
        if r:
            rows.append({"#": i + 1, "Name": it["name"], "Status": r["status"], "Time (ms)": r["ms"],
                         "Transformation": r["tech"], "Note": r.get("error", "")})
    return pd.DataFrame(rows)


def build_zip(progress=None) -> str:
    """Write transformed images chunk by chunk into a zip on disk."""
    s = st.session_state
    t = P.get(s.tech)
    params = params_for(t)
    path = os.path.join(data.workdir(), "results.zip")
    n = len(s.imgs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for i, it in enumerate(s.imgs):
            try:
                # Use stored processed image if available, otherwise reprocess
                if 'processed_images' in s and it["id"] in s.processed_images:
                    transformed = s.processed_images[it["id"]]
                else:
                    arr, _ = data.load_gray(it["path"], None)
                    transformed = t.func(arr, **params)

                buf = io.BytesIO()
                Image.fromarray(transformed, "L").save(buf, "PNG")
                stem = os.path.splitext(it["name"])[0]
                z.writestr(f"{i + 1:05d}_{stem}_processed_{t.key}.png", buf.getvalue())
            except Exception:
                pass
            if progress and (i % 5 == 0 or i + 1 == n):
                progress((i + 1) / n)
    return path
