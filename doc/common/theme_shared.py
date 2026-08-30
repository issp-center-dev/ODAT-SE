"""Shared Sphinx theme settings for ODAT-SE manuals."""

import os
import re


def get_version_info() -> tuple:
    """Return (version, release) read from src/odatse/_version.py.

    version is the short X.Y form, release is the full version string.
    The file is parsed with a regex rather than imported so that building
    the manuals does not require the package dependencies.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "src", "odatse", "_version.py")
    with open(path, encoding="utf-8") as f:
        release = re.search(
            r'__version__\s*=\s*["\']([^"\']+)["\']', f.read()
        ).group(1)
    version = ".".join(release.split("-")[0].split(".")[:2])
    return version, release


def get_theme_options() -> dict:
    return {
        "pygments_light_style": "a11y-high-contrast-light",
        "pygments_dark_style": "a11y-high-contrast-light",
        "navbar_align": "left",
        "navigation_depth": 3,
        "show_prev_next": True,
        "show_toc_level": 3,
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": ["navbar-icon-links", "search-field"],
        "secondary_sidebar_items": ["page-toc"],
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/issp-center-dev/ODAT-SE",
                "icon": "fa-brands fa-github",
            },
        ],
        "header_links_before_dropdown": 99,
        "navbar_persistent": [],
    }


HTML_CONTEXT = {
    "default_mode": "light",
}


HTML_CSS_FILES = ["custom.css", "css/custom.css"]
