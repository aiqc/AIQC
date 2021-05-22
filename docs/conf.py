# *** REFRESHING SIDEBAR MENU ***
# When you change menu/ page tiles. Run `make html` both with and without line above commented:
html_sidebars = { '**': ['globaltoc.html'] }


# -- Project information -----------------------------------------------------
project = 'AIQC'
copyright = '2020, Team AIQC'
author = 'Team AIQC'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom

# https://nbsphinx.readthedocs.io/en/0.8.0/
# These are pip packages. See `docs/requirements.txt` 
# Which is called by `.readthedocs.yml`
import sphinx_rtd_theme
extensions = [
	'nbsphinx'
	, 'sphinx_copybutton'
	, 'sphinx_rtd_theme'
	, 'sphinxext.opengraph'
]


# Adding this options due to plotly browser error w require.min.js
# https://cdnjs.com/libraries/require.js/2.1.10    Integrity in the "</> Copy Script Tag" button.
# https://github.com/readthedocs/sphinx_rtd_theme/issues/788#issuecomment-772785054
# also had to save `https://cdn.plot.ly/plotly-latest.min.js` to `/Users/layne/Desktop/aiqc/docs/_build/html/notebooks/plotly.js`

# nbsphinx_requirejs_options = {
# 	"src": "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
# 	"integrity": "sha512-VCK7oF67GXNc+J7zsu5o57jtxhLA75nSMHGaq8Q8TCOxDj4nMDw5dhQZvm9Cd9RN+3zgcodqbKcRc9gyPP8a2w==",
# 	"crossorigin": "anonymous"
# }


# https://nbsphinx.readthedocs.io/en/0.7.0/usage.html#suppress_warnings
suppress_warnings = [
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# nbsphinx automatically excludes '.ipynb_checkpoints'

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme" # see extension and `import` above.
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_css_files
# Create a CSS file at path: `.../docs/_static/css/custom.css`.
html_css_files = ['css/custom.css']
# See my css solution for svg: https://stackoverflow.com/questions/59215996/how-to-add-a-logo-to-my-readthedocs-logo-rendering-at-0px-wide
html_logo = 'images/aiqc_logo_wide_white_docs_300px.svg'
html_favicon = 'images/favicon.ico'
html_show_sphinx = False
html_show_copyright = False
html_title = 'AIQC'
html_short_title = 'AIQC'
html_theme_options = {
	'logo_only': True
	, 'display_version': False
	, 'sticky_navigation': False
	, 'collapse_navigation': False
}

hight_language = 'python3'

# For notebook hyperlinks, I seem to have to use a `[](.html)` syntax to get it to work.
# https://nbsphinx.readthedocs.io/en/0.8.1/markdown-cells.html#Links-to-Other-Notebooks

# `make html` is supposed to replicate to `.../docs/_build/html/_static/css/custom.css` 
# but I've been having to manually overwrite the _build css file at that location.

# -- <head><meta> title & links -----------------------------------------
# https://github.com/wpilibsuite/sphinxext-opengraph
# https://www.linkedin.com/pulse/how-clear-linkedin-link-preview-cache-ananda-kannan-p/
# https://www.linkedin.com/post-inspector/
ogp_site_url = "https://aiqc.readthedocs.io/"
ogp_site_name = "AIQC"
ogp_image = "https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/aiqc_logo_banner_controlroom.png"
ogp_image_alt = "Artificial Intelligence Quality Control"
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta property="twitter:image" content="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/images/aiqc_logo_banner_controlroom.png" />',
]
