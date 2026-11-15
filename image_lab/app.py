import streamlit as st

import data
import ui

st.set_page_config(page_title="Pixio", layout="wide", initial_sidebar_state="expanded")

ui.inject_css()
data.init_state()
ui.global_header()
ui.sidebar()
view = ui.compute_view()
ui.header(view)
ui.banner()

tab = st.session_state.get("tab") or "Workbench"
if tab == "Workbench":
    ui.workbench(view)
elif tab == "All results":
    ui.results_tab()
else:
    ui.about_tab()
ui.footer()
