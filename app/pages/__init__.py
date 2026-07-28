"""Streamlit pages: one module per screen, each exposing a single `render_*()` function.

`app/app.py` imports and calls these directly rather than relying on
Streamlit's filename-based auto-discovery, so page order, icons, and
titles stay under explicit control.
"""
