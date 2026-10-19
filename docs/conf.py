# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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
html_logo = "img/bopp_logo.svg"

html_theme_options = {
    "logo": {
        "alt_text": "bopp",
        "image_light": "img/bopp_logo.svg",
        "image_dark": "img/bopp_logo_dark.svg",
    },
    "show_version_warning_banner": True,
    "sticky_banners": True,
}
