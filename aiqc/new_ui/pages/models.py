# Local modules
import aiqc.orm
from aiqc.utils.meter import metrics_classify, metrics_regress
# UI modules
import dash
from dash import html, register_page

register_page(__name__)

# Dash naturally wraps in `id=_pages_content`
layout = html.Div(
    [
        html.P('models_page')
    ]
)