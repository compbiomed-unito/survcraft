"""Build the documentation from the checked-out SurvCraft sources."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "SurvCraft"
author = "Giovanni Birolo"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "README.md"]

autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

html_theme = "alabaster"
html_title = "SurvCraft documentation"
html_theme_options = {
    "github_user": "compbiomed-unito",
    "github_repo": "survcraft",
    "github_button": True,
}
