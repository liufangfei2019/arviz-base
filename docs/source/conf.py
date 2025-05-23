# pylint: disable=redefined-builtin,invalid-name
"""Sphinx configuration file."""

import os
from importlib.metadata import metadata

# -- Project information

_metadata = metadata("arviz-base")

project = _metadata["Name"]
author = _metadata["Author-email"].split("<", 1)[0].strip()
copyright = f"2022, {author}"

version = _metadata["Version"]
if os.environ.get("READTHEDOCS", None):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "jupyter_sphinx",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for extensions

extlinks = {
    "issue": ("https://github.com/arviz-devs/arviz-base/issues/%s", "GH#%s"),
    "pull": ("https://github.com/arviz-devs/arviz-base/pull/%s", "PR#%s"),
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "linkify"]

autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {
    "members": False,
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "optional", "scalar"}
singulars = ("int", "list", "dict", "float")
numpydoc_xref_aliases = {
    "DataArray": ":class:`xarray.DataArray`",
    "Dataset": ":class:`xarray.Dataset`",
    "DataTree": ":class:`xarray.DataTree`",
    **{f"{singular}s": f":any:`{singular}s <{singular}>`" for singular in singulars},
    "iterable": ":term:`python:iterable`",
    "sequence": ":term:`python:sequence`",
    "mapping": ":term:`python:mapping`",
    "hashable": ":term:`python:hashable`",
}

intersphinx_mapping = {
    "arviz_org": ("https://www.arviz.org/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/latest/", None),
    "arviz_plots": ("https://arviz-plots.readthedocs.io/en/latest", None),
    "arviz_stats": ("https://arviz-stats.readthedocs.io/en/latest", None),
}

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo": {
        "image_light": "_static/ArviZ.png",
        "image_dark": "_static/ArviZ_white.png",
    }
}
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "name.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}
