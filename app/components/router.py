"""
Customer Churn Prediction Platform
Application Router

Centralized routing system for the Streamlit application.

Responsibilities
----------------
• Register application pages
• Render navigation
• Execute selected routes

The router contains no business logic or UI rendering beyond
page selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

import streamlit as st
RenderFunction = Callable[[], None]

@dataclass(slots=True, frozen=True)
class Route:
    """
    Represents a navigable application page.
    """

    name: str
    title: str
    icon: str
    render: RenderFunction

    description: str = ""
    enabled: bool = True

    @property
    def label(self) -> str:
        """
        Navigation label displayed in the sidebar.
        """

        return f"{self.icon} {self.title}"

class RouteRegistry:
    """
    Stores and manages application routes.
    """

    def __init__(self) -> None:
        self._routes: dict[str, Route] = {}

    def register(
        self,
        route: Route,
    ) -> None:
        """
        Register a route.
        """

        if route.name in self._routes:
            raise ValueError(
                f"Route '{route.name}' is already registered."
            )

        self._routes[route.name] = route

    def get(
        self,
        name: str,
    ) -> Route:
        """
        Retrieve a registered route.
        """

        try:
            return self._routes[name]

        except KeyError as exc:
            raise KeyError(
                f"Unknown route: '{name}'."
            ) from exc

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a route exists.
        """

        return name in self._routes

    def routes(self) -> list[Route]:
        """
        Return enabled routes.
        """

        return [
            route
            for route in self._routes.values()
            if route.enabled
        ]

    def labels(self) -> list[str]:
        """
        Return sidebar labels.
        """

        return [
            route.label
            for route in self.routes()
        ]

    def execute(
        self,
        route: Route,
    ) -> None:
        """
        Execute a route.
        """

        route.render()

def render_navigation(
    registry: RouteRegistry,
) -> Route:
    """
    Render the sidebar navigation.

    Returns
    -------
    Route
        The selected application route.
    """

    routes = registry.routes()

    return st.sidebar.radio(
        label="Navigation",
        options=routes,
        format_func=lambda route: route.label,
    )

def build_router(
    routes: list[Route],
) -> RouteRegistry:
    """
    Build a populated route registry.
    """

    registry = RouteRegistry()

    for route in routes:
        registry.register(route)

    return registry

def execute_route(
    registry: RouteRegistry,
    route_name: str,
) -> None:
    """
    Backward-compatible execution helper.

    Existing imports can continue using this function
    without modification.
    """

    registry.execute(
        registry.get(route_name),
    )

__all__: Final = [
    "Route",
    "RouteRegistry",
    "build_router",
    "render_navigation",
    "execute_route",
]