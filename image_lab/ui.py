"""All rendering. Each function draws one region of the page."""
from __future__ import annotations

import html
import io
import math
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import batch
import data
import processing as P

TABS = ["Workbench", "All results", "About"]
LOGO = ('<svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="#6C9BD1" stroke-width="1.6" aria-hidden="true">'
        '<rect x="2" y="2" width="16" height="16" rx="2"/><path d="M2 14L8 8l4 4 6-6"/></svg>')


def md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def inject_css():
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    md(f"<style>{css}</style>")


def thumb_css(rules: dict):
    """Per-button background images (used for Browse strip and Grid tiles)."""
    css = "".join(f'.st-key-{k} button{{background-image:url("{u}")}}' for k, u in rules.items())
    md(f"<style>{css}</style>")


def sec(title: str, note: str = "", css_class: str = ""):
    class_attr = f' class="sec {css_class}"' if css_class else ' class="sec"'
    md(f'<div{class_attr}><span>{title}</span><small>{note}</small></div>')


def global_header():
    """Unified header spanning full width of the application."""
    md(f'<div class="global-header">{LOGO}Pixio</div>')


# -------------------------------------------------------------- callbacks ---
def cb_tech(key):
    st.session_state.tech = key


def cb_select(i):
    data.select(i)


def cb_remove(item_id):
    data.remove(item_id)


def cb_step(d):
    data.select(st.session_state.cur + d)


def cb_page(d, total_pages):
    st.session_state.page = max(0, min(st.session_state.page + d, total_pages - 1))


def cb_grid_page(d, pages):
    st.session_state.grid_page = max(0, min(st.session_state.grid_page + d, pages - 1))


def cb_open_row():
    sel = st.session_state.get("res_table")
    rows = sel["selection"]["rows"] if sel else []
    if rows:
        df = batch.table()
        if rows[0] < len(df):
            data.select(int(df.iloc[rows[0]]["#"]) - 1)
            st.session_state.tab = "Workbench"


# ---------------------------------------------------------------- sidebar ---
def sidebar():
    s = st.session_state
    with st.sidebar:
        sec("Transformation", css_class="sec-transformation")
        with st.container(key="tlist"):
            for group in ("Intensity", "Contrast"):
                md(f'<div class="grp" style="padding:12px 12px 4px 12px;border-top:1px solid var(--b2);border-bottom:1px solid var(--b2);margin-top:8px; margin-bottom:20px; padding-top:8px;padding-bottom:8px">{group}</div>')
                for t in P.TRANSFORMS:
                    if t.group == group:
                        st.button(t.name, key=f"t_{t.key}", on_click=cb_tech, args=(t.key,),
                                  type="primary" if s.tech == t.key else "secondary")

        sec("Equation", css_class="sec-equation")
        t = P.get(s.tech)
        md(f'<div class="eq">{t.formula}</div>')
        for v in t.vars:
            is_int = float(v.step).is_integer()
            cast = int if is_int else float
            st.slider(v.label, cast(v.min), cast(v.max), cast(v.default), cast(v.step), key=f"v_{t.key}_{v.key}")

        n = len(s.imgs)
        sec("Input", f"{n:,} image{'s' if n != 1 else ''}", css_class="sec-input")
        src = st.segmented_control("Source", ["Upload", "Hugging Face"], default="Upload", key="src",
                                   label_visibility="collapsed") or "Upload"
        if src == "Hugging Face":
            hf_form()

        if n:
            image_list()
        add_zone(n)


def hf_form():
    st.text_input("Dataset ID", key="hf_id", placeholder="user/dataset-name")
    c1, c2 = st.columns(2)
    c1.text_input("Split", value="train", key="hf_split")
    c2.number_input("Limit", min_value=1, max_value=100000, value=1000, step=100, key="hf_limit")
    st.checkbox("Stream instead of downloading all", value=True, key="hf_stream")
    if st.button("Load images", key="hf_load"):
        ds_id = (st.session_state.hf_id or "").strip()
        if not ds_id:
            st.warning("Enter a dataset ID, e.g. user/dataset-name.")
            return
        bar = st.progress(0.0, text="Connecting…")
        try:
            added = data.load_hf(ds_id, st.session_state.hf_split.strip() or "train", int(st.session_state.hf_limit),
                                 st.session_state.hf_stream,
                                 lambda i, tot: bar.progress(min(i / tot, 1.0), text=f"Loading {i:,} / {tot:,}"))
        except Exception as e:
            bar.empty()
            st.error(f"Could not load dataset: {e}")
            return
        data.select(st.session_state.cur)
        st.toast(f"Added {added:,} images")
        st.rerun()


def image_list():
    s = st.session_state
    items = s.imgs
    n = len(items)
    idx = list(range(n))
    if n > 10:
        q = st.text_input("Filter", key="filter", placeholder="Filter by name or label…",
                          label_visibility="collapsed").strip().lower()
        if q:
            idx = [i for i in idx if q in items[i]["name"].lower() or q in (items[i]["label"] or "").lower()]
    ps = data.page_size()
    pages = max(1, math.ceil(len(idx) / ps))
    page = max(0, min(s.page, pages - 1))
    s.page = page
    shown = idx[page * ps:(page + 1) * ps]
    if not shown:
        md('<div class="hint">No images match.</div>')

    rules = {}
    with st.container(key="ilist"):
        for i in shown:            # i = index in the FULL list
            it = items[i]
            k = f"sel_{it['id']}"
            rules[k] = data.thumb_uri(it["path"], 48)
            c1, c2 = st.columns([8, 1], vertical_alignment="center")
            c1.button(it["name"], key=k, on_click=cb_select, args=(i,),
                      type="primary" if i == s.cur else "secondary",
                      help=f"{it['name']} · {it['label'] or data.human(it['size'])}")
            c2.button("×", key=f"rm_{it['id']}", on_click=cb_remove, args=(it["id"],), help=f"Remove {it['name']}")
    thumb_css(rules)
    if n > 10:
        c1, c2, c3 = st.columns([1, 3, 1], vertical_alignment="center")
        c1.button("‹", key="pg_prev", on_click=cb_page, args=(-1, pages), disabled=page == 0)
        c2.markdown(f'<div class="navlbl hint">Page {page + 1:,} of {pages:,}</div>', unsafe_allow_html=True)
        c3.button("›", key="pg_next", on_click=cb_page, args=(1, pages), disabled=page >= pages - 1)


def add_zone(n: int):
    s = st.session_state
    with st.container(key="drop_empty" if n == 0 else "drop_more"):
        files = st.file_uploader("Add images", type=data.EXTS, accept_multiple_files=True,
                                 key=f"up_{s.up_gen}", label_visibility="collapsed")
    if files:
        data.add_files(files)
        s.up_gen += 1
        data.select(s.cur)
        st.rerun()


# ----------------------------------------------------------------- header ---
def compute_view():
    s = st.session_state
    if not s.imgs:
        return None
    it = s.imgs[max(0, min(s.cur, len(s.imgs) - 1))]
    t = P.get(s.tech)
    params = batch.params_for(t)
    try:
        arr, size = data.load_gray_cached(it["path"])
        out = t.func(arr, **params)
    except Exception as e:
        return {"error": str(e), "item": it}
    return {"item": it, "arr": arr, "out": out, "size": size, "tech": t, "params": params}


def header(view):
    s = st.session_state
    n = len(s.imgs)
    done = sum(1 for r in s.results.values() if r["status"] == "done")

    def fmt(tab):
        if tab == "All results" and n > 1 and s.results:
            return f"All results · {done:,}" + (f" / {n:,}" if done != n else "")
        return tab

    with st.container(key="topbar"):
        c = st.columns([10, 1.4, 2.2], vertical_alignment="center")
        with c[0]:
            st.segmented_control("Page", TABS, key="tab", format_func=fmt, label_visibility="collapsed")

        # Download button: if results exist, download zip of all processed images
        has_results = bool(s.results and done > 0)
        download_data = b""
        download_name = "result.png"
        download_mime = "image/png"
        download_disabled = True

        if has_results:
            # Build zip with all processed images
            if not s.zip_path:
                s.zip_path = batch.build_zip()
            try:
                with open(s.zip_path, "rb") as f:
                    download_data = f.read()
                download_name = "processed_images.zip"
                download_mime = "application/zip"
                download_disabled = False
            except OSError:
                pass
        elif view and "out" in view:
            # Single image download (current behavior)
            buf = io.BytesIO()
            Image.fromarray(view["out"], "L").save(buf, "PNG")
            download_data = buf.getvalue()
            download_name = f"{view['item']['name'].rsplit('.', 1)[0]}_processed_{view['tech'].key}.png"
            download_disabled = False

        with c[1]:
            # Change button label based on what's available
            dl_label = "Download ZIP" if has_results else "Download"
            st.download_button(dl_label, download_data, file_name=download_name, mime=download_mime,
                             disabled=download_disabled, key="dl")
        with c[2]:
            label = "Process image" if n <= 1 else f"Process all {n:,}"
            running = bool(s.job and s.job["active"])
            st.button(label, key="process", type="primary", on_click=batch.start_job, disabled=n == 0 or running)


# ----------------------------------------------------------- batch banner ---
def banner():
    s = st.session_state
    job = s.job
    if not job or not job["active"]:
        return
    runner = st.fragment(run_every=0.6)(_banner_body)
    runner()


def _banner_body():
    s = st.session_state
    job = s.job
    finished = batch.step()
    total = len(job["ids"])
    pct = 100 * job["pos"] / max(total, 1)
    el = max(time.time() - job["t0"], 1e-6)
    rate = job["pos"] / el
    left = (total - job["pos"]) / rate if rate else 0
    if finished:
        st.rerun()
    with st.container(key="banner"):
        c = st.columns([2.2, 5, 2.6, 1.1], vertical_alignment="center")
        c[0].markdown(f'<div class="banner">Processing {job["pos"]:,} / {total:,}</div>', unsafe_allow_html=True)
        c[1].markdown(f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>', unsafe_allow_html=True)
        c[2].markdown(f'<div class="meta">{rate:,.0f} images/s · {left:,.0f} s left</div>', unsafe_allow_html=True)
        c[3].button("Cancel", key="cancel", on_click=batch.cancel_job)



# -------------------------------------------------------------- workbench ---
def panel(title, uri, right=""):
    return (f'<div class="pn"><div class="pt"><span>{title}</span><span>{right}</span></div>'
            f'<div class="vp"><img src="{uri}" alt="{title}"></div></div>')


def hist_panel(title, arr, ymax, color):
    h = np.bincount(arr.ravel(), minlength=256).astype(float).reshape(64, 4).sum(1)
    W, H, bw = 280, 170, 280 / 64
    bars = "".join(
        f'<rect x="{i * bw:.2f}" y="{H - v / ymax * H * .96:.2f}" width="{bw - .6:.2f}" height="{v / ymax * H * .96:.2f}" fill="{color}"/>'
        for i, v in enumerate(h))
    return (f'<div class="pn"><div class="pt"><span>{title}</span></div><div class="hp">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">{bars}</svg>'
            f'<div class="ax"><span>0</span><span>128</span><span>255</span></div></div></div>')


def nav(view):
    s = st.session_state
    n, cur = len(s.imgs), s.cur
    grid_mode = (s.get("mode") or "Single") == "Grid"
    c = st.columns([1.0], vertical_alignment="center")
    if grid_mode:
        c[0].markdown(f'<div class="navlbl">{n:,} image{"s" if n != 1 else ""}</div>', unsafe_allow_html=True)
    else:
        c[0].markdown('', unsafe_allow_html=True)


def empty_state():
    md('<div class="empty"><b>No images yet</b><span>Drop images in the Input section on the left, '
       'or load some from the Hugging Face tab.</span></div>')


def grid_with_header(t, params):
    """Grid view with header showing page info and mode toggle."""
    s = st.session_state
    items, n = s.imgs, len(s.imgs)
    per = 12
    pages = max(1, math.ceil(n / per))
    gp = max(0, min(s.grid_page, pages - 1))
    s.grid_page = gp
    ptuple = tuple(sorted(params.items()))

    # Grid bar with card styling
    with st.container(key="viewer"):
        h = st.columns([8, 3], vertical_alignment="center")
        h[0].markdown(f'<div class="vh"><b>{n:,} image{"s" if n != 1 else ""}</b><span>{t.name}</span></div>', unsafe_allow_html=True)
        with h[1]:
            st.segmented_control("Mode", ["Single", "Grid"], default="Single", key="mode", label_visibility="collapsed")

    # Create grid cards using the same panel helper
    cards = []
    for it in items[gp * per:(gp + 1) * per]:
        o, r = data.card_uris(it["path"], t.key, ptuple)
        name = html.escape(it["name"])
        cards.append(
            f'<div class="gcard">'
            f'<div class="vh"><b title="{name}">{name}</b></div>'
            f'<div class="pair">'
            f'{panel("Original", o)}{panel("Result", r)}'
            f'</div></div>')
    md('<div class="cards">' + "".join(cards) + "</div>")

    # Navigation controls
    if pages > 1:
        c = st.columns([1, 3, 1], vertical_alignment="center")
        c[0].button("‹", key="gp_prev", on_click=cb_grid_page, args=(-1, pages), disabled=gp == 0)
        c[1].markdown(f'<div class="navlbl hint">Page {gp + 1:,} of {pages:,}</div>', unsafe_allow_html=True)
        c[2].button("›", key="gp_next", on_click=cb_grid_page, args=(1, pages), disabled=gp >= pages - 1)


def workbench(view):
    s = st.session_state
    if not s.imgs:
        empty_state()
        return
    if view is None or "error" in view:
        st.error(f"Could not read this image: {view.get('error') if view else ''}")
        return
    t, params = view["tech"], view["params"]
    mode = s.get("mode") or "Single"
    if mode == "Grid":
        grid_with_header(t, params)
        return
    arr, out = view["arr"], view["out"]

    # Create big card with file title and two small image cards
    file_name = view["item"]["name"]
    n, cur = len(s.imgs), s.cur

    # Viewer bar card with header only
    with st.container(key="viewer"):
        h = st.columns([5, 0.6, 0.6, 2], vertical_alignment="center")
        h[0].markdown(f'<div class="vh"><b>{file_name}</b><span>{cur + 1} of {n}</span></div>', unsafe_allow_html=True)
        h[1].button("‹", key="prev", on_click=cb_step, args=(-1,), disabled=cur <= 0)
        h[2].button("›", key="next", on_click=cb_step, args=(1,), disabled=cur >= n - 1)
        with h[3]:
            st.segmented_control("Mode", ["Single", "Grid"], default=mode, key="mode", label_visibility="collapsed")

    # Image card (separate from bar)
    md(f'<div class="gcard single"><div class="pair">'
       f'{panel("Original", data.to_uri(arr))}'
       f'{panel(f"Result · {t.name}", data.to_uri(out))}'
       f'</div></div>')

    if len(s.imgs) > 10:
        browse()
    if t.shows_histogram:
        a = np.bincount(arr.ravel(), minlength=256).astype(float).reshape(64, 4).sum(1)
        b = np.bincount(out.ravel(), minlength=256).astype(float).reshape(64, 4).sum(1)
        ymax = max(a.max(), b.max(), 1)
        md('<div class="pair">' + hist_panel("Histogram · Original", arr, ymax, "#8A8A90")
           + hist_panel("Histogram · Equalized", out, ymax, "#6C9BD1") + "</div>")


def browse():
    s = st.session_state
    items, n, cur = s.imgs, len(s.imgs), s.cur
    W = 12
    start = max(0, min(cur - W // 2, n - W))
    idxs = list(range(start, min(n, start + W)))
    rules = {}
    with st.container(key="browse"):
        md(f'<div class="pt"><span>Browse</span><span>Showing {idxs[0] + 1:,}–{idxs[-1] + 1:,} of {n:,}</span></div>')
        cols = st.columns(W)
        for col, i in zip(cols, idxs):
            it = items[i]
            k = f"bt_{it['id']}"
            rules[k] = data.thumb_uri(it["path"], 72)
            col.button(str(i + 1), key=k, on_click=cb_select, args=(i,), type="primary" if i == cur else "secondary")
    thumb_css(rules)


# ---------------------------------------------------------- other tabs ------
def results_tab():
    s = st.session_state
    df = batch.table()
    if df.empty:
        md('<div class="empty"><b>Nothing processed yet</b><span>Use the button in the header to run the selected '
           'transformation on your images.</span></div>')
        return
    done = int((df["Status"] == "done").sum())
    failed = int((df["Status"] == "failed").sum())
    mean = df["Time (ms)"].dropna().mean()
    md(f'<div class="meta">{done:,} done · {failed:,} failed · mean {mean:.2f} ms per image · '
       f'{df["Transformation"].iloc[0]}</div>')
    st.dataframe(df, hide_index=True, width="stretch", on_select=cb_open_row,
                 selection_mode="single-row", key="res_table", height=420)
    c = st.columns([1.3, 1.6, 1.6, 6])
    c[0].download_button("Export CSV", df.to_csv(index=False).encode(), "image_lab_results.csv", "text/csv")
    if c[1].button("Prepare ZIP", key="zip_btn"):
        bar = st.progress(0.0, text="Writing images…")
        s.zip_path = batch.build_zip(lambda p: bar.progress(p, text=f"Writing images… {p:.0%}"))
        bar.empty()
    if s.zip_path:
        try:
            with open(s.zip_path, "rb") as f:
                c[2].download_button("Download ZIP", f, "image_lab_results.zip", "application/zip", key="zip_dl")
        except OSError:
            s.zip_path = None


def about_tab():
    md("""<div class="pn" style="padding:16px 18px;color:var(--tx2);line-height:1.6;max-width:760px">
<b style="color:var(--tx)">Pixio</b> · COM 508 Advanced Digital Image Processing<br><br>
Pick a transformation, adjust its variables, and compare the result with the original. Works with a few uploaded
images or with thousands streamed from the Hugging Face Hub. Images are stored on disk and decoded on demand,
so memory stays flat. Use <i>Process all</i> to run the technique over every image; export the table as CSV
or the transformed images as a ZIP.</div>""")


def footer():
    md('<div class="footer"><span>COM 508 · Advanced Digital Image Processing</span><span>Sondos Deeb · Nov 2026</span></div>')
