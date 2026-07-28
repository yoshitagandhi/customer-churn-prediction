"""
Customer Churn Prediction Platform
Navigation Registry

Centralized registry containing application page metadata.

Responsibilities
----------------
• Register application pages
• Provide page lookup utilities
• Control page ordering
• Define the default landing page

This module contains metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

@dataclass(slots=True, frozen=True)
class Page:
    """
    Represents a registered application page.
    """

    id: str
    title: str
    icon: str
    path: str
    description: str

    show_in_sidebar: bool = True

    @property
    def label(self) -> str:
        """
        Sidebar label.
        """

        return f"{self.icon} {self.title}"

PAGES: Final[tuple[Page, ...]] = (
    # Keep your existing Page(...) declarations here unchanged
)

DEFAULT_PAGE_ID: Final[str] = "dashboard"

_PAGE_LOOKUP: Final[dict[str, Page]] = {
    page.id: page
    for page in PAGES
}

def get_navigation() -> tuple[Page, ...]:
    """
    Return pages in display order.
    """

    return PAGES


def get_sidebar_pages() -> tuple[Page, ...]:
    """
    Return sidebar pages.
    """

    return tuple(
        page
        for page in PAGES
        if page.show_in_sidebar
    )


def get_default_page() -> Page:
    """
    Return the application's default page.
    """

    return _PAGE_LOOKUP[DEFAULT_PAGE_ID]


def get_page(
    page_id: str,
) -> Page:
    """
    Retrieve a page by identifier.
    """

    try:
        return _PAGE_LOOKUP[page_id]

    except KeyError as exc:
        raise KeyError(
            f"Unknown page: '{page_id}'."
        ) from exc


def page_exists(
    page_id: str,
) -> bool:
    """
    Return whether a page exists.
    """

    return page_id in _PAGE_LOOKUP


def page_titles() -> tuple[str, ...]:
    """
    Return page titles.
    """

    return tuple(
        page.title
        for page in PAGES
    )


def page_ids() -> tuple[str, ...]:
    """
    Return registered page identifiers.
    """

    return tuple(
        _PAGE_LOOKUP.keys()
    )


def sidebar_labels() -> tuple[str, ...]:
    """
    Return sidebar labels.
    """

    return tuple(
        page.label
        for page in get_sidebar_pages()
    )

__all__ = [
    "Page",
    "PAGES",
    "DEFAULT_PAGE_ID",
    "get_navigation",
    "get_sidebar_pages",
    "get_default_page",
    "get_page",
    "page_exists",
    "page_titles",
    "page_ids",
    "sidebar_labels",
]