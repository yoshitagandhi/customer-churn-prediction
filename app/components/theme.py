"""
Customer Churn Prediction Platform
Theme System

Centralized design system used across the Streamlit application.

Responsibilities
----------------
• Design tokens
• CSS generation
• Theme application
• Color utilities
• Layout styling

This module contains presentation logic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import streamlit as st

@dataclass(frozen=True)
class ColorPalette:
    """Application color system."""

    primary: str = "#2563EB"
    secondary: str = "#0F172A"

    success: str = "#16A34A"
    warning: str = "#EAB308"
    error: str = "#DC2626"
    info: str = "#0284C7"

    background: str = "#F8FAFC"
    surface: str = "#FFFFFF"

    text_primary: str = "#0F172A"
    text_secondary: str = "#64748B"

    border: str = "#E2E8F0"


@dataclass(frozen=True)
class Typography:
    """Typography tokens."""

    font_family: str = (
        "Inter, Segoe UI, sans-serif"
    )

    title_size: str = "2rem"
    header_size: str = "1.4rem"
    body_size: str = "0.95rem"


@dataclass(frozen=True)
class Layout:
    """Layout tokens."""

    border_radius: str = "14px"

    content_padding: str = "1rem"


COLORS: Final = ColorPalette()

TYPOGRAPHY: Final = Typography()

LAYOUT: Final = Layout()

@dataclass(frozen=True)
class Theme:
    """Combined theme tokens."""

    colors: ColorPalette = COLORS
    typography: Typography = TYPOGRAPHY
    layout: Layout = LAYOUT

THEME: Final = Theme()

def build_root_variables() -> str:
    """
    Build global CSS variables from the application theme.
    """
    colors = THEME.colors
    typography = THEME.typography
    layout = THEME.layout

    return f"""
    :root {{

        /* Colors */
        --primary: {colors.primary};
        --secondary: {colors.secondary};

        --success: {colors.success};
        --warning: {colors.warning};
        --error: {colors.error};
        --info: {colors.info};

        --background: {colors.background};
        --surface: {colors.surface};

        --text-primary: {colors.text_primary};
        --text-secondary: {colors.text_secondary};

        --border: {colors.border};

        /* Typography */
        --font-family: {typography.font_family};

        --title-size: {typography.title_size};
        --header-size: {typography.header_size};
        --body-size: {typography.body_size};

        /* Layout */
        --radius: {layout.border_radius};
        --content-padding: {layout.content_padding};

    }}
    """
    
def build_card_css() -> str:
    """
    Build reusable dashboard card styling.
    """

    return """
    .dashboard-card {

        background: var(--surface);

        border: 1px solid var(--border);

        border-radius: var(--radius);

        padding: var(--content-padding);

        margin-bottom: 1rem;

    }
    """
    
def build_button_css() -> str:
    """
    Build Streamlit button styling.
    """

    return """
    .stButton button {

        border-radius: 12px;

        font-family: var(--font-family);

        font-weight: 600;

        transition:
            background-color 0.2s ease,
            color 0.2s ease,
            transform 0.15s ease;

    }

    .stButton button:hover {

        transform: translateY(-1px);

    }
    """
    
def build_sidebar_css() -> str:
    """
    Build sidebar styling.
    """

    return """
    section[data-testid="stSidebar"] {

        border-right: 1px solid var(--border);

        background: var(--surface);

    }
    """
    
def build_layout_css() -> str:
    """
    Build shared application layout styling.
    """

    return """
    .main {

        background: var(--background);

        color: var(--text-primary);

        font-family: var(--font-family);

    }

    h1 {

        font-size: var(--title-size);

    }

    h2 {

        font-size: var(--header-size);

    }

    p {

        font-size: var(--body-size);

        color: var(--text-secondary);

    }
    """
    
def build_theme() -> str:
    """
    Assemble the complete application stylesheet.
    """

    builders = (
        build_root_variables,
        build_layout_css,
        build_card_css,
        build_button_css,
        build_sidebar_css,
    )

    return "\n".join(
        builder()
        for builder in builders
    )

class ThemeManager:
    """
    Coordinates theme generation and application.

    This class is intentionally lightweight—it provides a
    single entry point for working with the application's
    visual design system.
    """

    @staticmethod
    def build() -> str:
        """
        Build the complete application stylesheet.
        """

        return build_theme()

    @staticmethod
    def apply() -> None:
        """
        Inject the application stylesheet into Streamlit.
        """

        st.markdown(
            f"""
            <style>

            {ThemeManager.build()}

            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def color(name: str) -> str:
        """
        Retrieve a named application color.

        Raises
        ------
        KeyError
            If the requested color does not exist.
        """

        palette = THEME.colors

        colors = {
            "primary": palette.primary,
            "secondary": palette.secondary,
            "success": palette.success,
            "warning": palette.warning,
            "error": palette.error,
            "info": palette.info,
            "background": palette.background,
            "surface": palette.surface,
            "text_primary": palette.text_primary,
            "text_secondary": palette.text_secondary,
            "border": palette.border,
        }

        return colors[name]

def apply_theme() -> None:
    """
    Apply the application theme.
    """

    ThemeManager.apply()


def get_color(
    name: str,
) -> str:
    """
    Retrieve a theme color.
    """

    return ThemeManager.color(name)

__all__ = [
    "THEME",
    "ThemeManager",
    "apply_theme",
    "get_color",
]