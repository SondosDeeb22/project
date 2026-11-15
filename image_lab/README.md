# Image Lab

Streamlit workbench for testing digital image processing techniques on a few uploaded images
or on large Hugging Face datasets.

## Run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501

## Files
- `app.py`        page assembly
- `ui.py`         every region of the page (sidebar, header, viewer, browse, grid, results)
- `processing.py` transformations + registry (formula, variables, histogram flag). Add a technique = one function + one entry.
- `data.py`       collection (add/remove), Hugging Face loading, lazy decode, caches
- `batch.py`      chunked "Process all" runner, CSV/ZIP export
- `styles.css`    all styling (dark theme tokens)
- `.streamlit/config.toml` theme

If you already have your own transformation code, paste it into `processing.py` (uint8 gray array in, uint8 out).
