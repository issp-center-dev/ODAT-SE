"""Shared Sphinx theme settings for ODAT-SE manuals."""


def get_theme_options() -> dict:
    return {
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
