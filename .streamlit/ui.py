"""
===============================================================================
Customer Churn Prediction Platform
Unified UI API

File        : ui.py
Version     : 1.0

Purpose
-------
Public interface for the Streamlit UI framework.

This module re-exports the functionality provided by the
theme, layout, components, animations, and icons modules,
allowing application code to import from a single location.

Example
-------
from streamlit_ui.ui import *

initialize_theme()

page_header(
    title="Dashboard",
    subtitle="Customer Churn Prediction",
    icon=DASHBOARD,
)

metric_card(
    title="ROC-AUC",
    value="0.942",
)

with loading("Running prediction..."):
    ...
===============================================================================
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent


def _local_module(name: str):
    module_path = _current_dir / f"{name}.py"
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(name, str(module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(name)
theme = _local_module("theme")

horizontal_rule = theme.horizontal_rule
initialize_theme = theme.initialize_theme
inject_html = theme.inject_html
load_css = theme.load_css
page_break = theme.page_break

layout = _local_module("layout")

card = layout.card
centered_container = layout.centered_container
divider = layout.divider
empty_state = layout.empty_state
end_container = layout.end_container
footer = layout.footer
four_columns = layout.four_columns
hero = layout.hero
page_header = layout.page_header
section = layout.section
section_title = layout.section_title
spacer = layout.spacer
three_columns = layout.three_columns
two_columns = layout.two_columns

_components = _local_module("components")
globals().update(
    {
        name: getattr(_components, name)
        for name in dir(_components)
        if not name.startswith("_")
    }
)

animations = _local_module("animations")

animated_progress = animations.animated_progress
error = animations.error
fade_in = animations.fade_in
info = animations.info
loading = animations.loading
skeleton = animations.skeleton
status_transition = animations.status_transition
success = animations.success
toast = animations.toast
warning = animations.warning

_icons = _local_module("icons")
globals().update(
    {
        name: getattr(_icons, name)
        for name in dir(_icons)
        if name.isupper()
    }
)

__all__ = [
    # Theme
    "initialize_theme",
    "load_css",
    "inject_html",
    "page_break",
    "horizontal_rule",

    # Layout
    "page_header",
    "hero",
    "section_title",
    "divider",
    "spacer",
    "section",
    "card",
    "empty_state",
    "footer",
    "two_columns",
    "three_columns",
    "four_columns",
    "centered_container",
    "end_container",

    # Animations
    "loading",
    "fade_in",
    "skeleton",
    "toast",
    "success",
    "error",
    "warning",
    "info",
    "animated_progress",
    "status_transition",
]

# Export everything from components.py
__all__.extend(
    name
    for name in globals()
    if callable(globals()[name])
    and name not in __all__
)

# Export all icon constants
__all__.extend(
    name
    for name in globals()
    if name.isupper()
)