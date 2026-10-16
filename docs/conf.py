# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bopp'
copyright = '2026, Brian McFee'
author = 'Brian McFee'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.doctest',
        'sphinx.ext.viewcode',
        'sphinx.ext.intersphinx',
        'numpydoc',
        "myst_parser",
]

myst_enable_extensions = ["colon_fence", "attrs_inline", "linkify", "tasklist"]

templates_path = ['templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'templates/bopp-extension-template']

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'msgspec': ('https://msgspec.dev/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'polars': ('https://docs.pola.rs/py-polars/html/', None),
    'pyarrow': ('https://arrow.apache.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']


def _create_extension_template_archive(app):
    """Package the minimal extension template directory into a downloadable zip file."""
    docs_dir = Path(app.srcdir)
    template_dir = docs_dir / "templates" / "bopp-extension-template"
    static_dir = docs_dir / "_static"

    if template_dir.is_dir():
        static_dir.mkdir(parents=True, exist_ok=True)
        archive_base = static_dir / "bopp-extension-template"
        shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(template_dir),
        )


def setup(app):
    app.connect("builder-inited", _create_extension_template_archive)
html_logo = "img/bopp_logo.svg"

html_theme_options = {
    "logo": {
        "alt_text": "bopp",
        "image": html_logo
    },
    "show_version_warning_banner": True,
    "sticky_banners": True,
}
