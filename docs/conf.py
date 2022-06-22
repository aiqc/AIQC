"""
	<!> REFRESHING SIDEBAR MENU <!>
	When you change menu/ file names, you need to run `make html` 
	with this line uncommented, and then run it again with it commented:
"""
#html_sidebars = { '**': ['globaltoc.html'] }

project   = 'AIQC'
copyright = '2020, Team AIQC'
author    = 'Team AIQC'

"""
	====== PLUGINS ======
	https://nbsphinx.readthedocs.io/en/0.8.0/
	These are pip packages. See `docs/requirements.txt` 
	Which is called by `.readthedocs.yml`
	https://nbsphinx.readthedocs.io/en/0.7.0/usage.html#suppress_warnings
"""
import sphinx_rtd_theme
extensions = [
	'nbsphinx'
	, 'sphinx_copybutton'
	, 'sphinx_rtd_theme'
	, 'sphinxext.opengraph'
	, 'sphinxcontrib.youtube'
]
suppress_warnings  = ['nbsphinx']
highlight_language = 'python3'

"""
	====== ASSETS ======
	Acts like .gitignore. nbsphinx automatically excludes '.ipynb_checkpoints'
"""
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'Untitled.ipynb']
templates_path   = ['_templates']
html_static_path = ['_static']
html_css_files   = ['css/custom.css']
html_js_files    = []

"""
	====== OPTIONS ======
	SVG logo: stackoverflow.com/questions/59215996/how-to-add-a-logo-to-my-readthedocs-logo-rendering-at-0px-wide
"""

html_theme          = "sphinx_rtd_theme" # see extension and `import` above.
html_logo           = '_static/images/web/aiqc_logo_square_blues_transparent.svg'
html_favicon        = '_static/images/web/favicon.ico'
html_show_sphinx    = False
html_show_copyright = False
html_title          = 'AIQC'
html_short_title    = 'AIQC'
html_theme_options  = dict(
	logo_only                    = True
	, display_version            = False
	, sticky_navigation          = False
	, collapse_navigation        = False
    , prev_next_buttons_location = None
)

"""
	====== HTML HEAD & META ======
	https://github.com/wpilibsuite/sphinxext-opengraph
	https://www.linkedin.com/pulse/how-clear-linkedin-link-preview-cache-ananda-kannan-p/
	https://www.linkedin.com/post-inspector/
"""
ogp_site_url         = "https://docs.aiqc.io/"
ogp_site_name        = "AIQC"
ogp_image            = "https://raw.githubusercontent.com/aiqc/aiqc/main/docs/_static/images/web/meta_image_tall_rect.png"
ogp_image_alt        = "Artificial Intelligence Quality Control"
ogp_type             = "website"
ogp_custom_meta_tags = [
    '<meta property="twitter:image" content="https://raw.githubusercontent.com/aiqc/aiqc/main/docs/_static/images/web/meta_image_tall_rect.png" />'
]
