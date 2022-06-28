# Local modules
from aiqc import orm
# UI modules
from dash import register_page, html, dcc, callback
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np


register_page(__name__)

refresh_seconds = 10*1000


# Dash naturally wraps in `id=_pages_content`
layout = html.Div(
    [
        dcc.Interval(
            id="initial_load",
            n_intervals=0,
            max_intervals=-1, 
            interval=refresh_seconds
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Model ID"),
                                # Not a callback because it is the first inputable object.
                                dbc.Select(id='pred_dropdown'),
                            ],
                            size="sm", className='ctrl_chart ctrl_big ctr'
                        ),
                        html.Div(id='sim-features')
                    ],
                    width='4', align='center', className='sim-inputs'
                ),
                dbc.Col(
                    [
                        'right'
                    ],
                    width='8', align='center', className='sim-outputs'
                ),
            ],
            className='sim-pane'
        )
    ]
)


"""
- When the page is started and refreshed, this updates things.
- You can only use objects at the start of the DAG, 
  e.g. not `exp_plot.clickData` because it doesn't exist yet.
"""
@callback(
    [
        Output(component_id="pred_dropdown", component_property="options"),
        Output(component_id="pred_dropdown", component_property="placeholder"),
        Output(component_id="pred_dropdown", component_property="value"),        
    ],
    Input(component_id="initial_load",  component_property="n_intervals"),
    State(component_id='pred_dropdown', component_property='value'),
)
def refresh_predictors(n_intervals:int, model_id:int):
    models = list(orm.Predictor)
    if (not models):
        return [], "None yet", []
    
    models.reverse()
    model_options = []
    for m in models:
        label = f"Model:{m.id}"
        if (m.is_starred):
            label += " ★"
        opt = dict(label=label, value=m.id)
        model_options.append(opt)
    
    if (model_id is None):
        model_id = models[0].id
    # else:
    #     raise PreventUpdate
    return model_options, "Select model", model_id


@callback(
    Output(component_id='sim-features', component_property='children'),
    Input(component_id='pred_dropdown', component_property='value')
)
def populate_features(model_id:int):
    model = orm.Predictor.get_by_id(model_id)
    features = model.job.queue.splitset.features
    
    kids = []
    for f in features:
        typ = f.dataset.typ
        # Need to check the type, so don't capitalize yet.
        subtitle = f"Feature: {typ.capitalize()}"
        subtitle = html.P(subtitle, className='sim-subtitle')
        kids.append(subtitle)

        if (typ=='tabular'):
            stats_numeric   = f.dataset.stats_numeric
            stats_categoric = f.dataset.stats_categoric

            f_typs = f.get_dtypes()
            for col, typ in f_typs.items():
                is_numeric = np.issubdtype(typ, np.number)
                is_date    = np.issubdtype(typ, np.datetime64)
                
                if (is_numeric or is_date):
                    stats   = stats_numeric[col]
                    minimum = stats['min']
                    maximum = stats['max']
                    mean    = stats['mean']

                    field = html.Div(
                        [
                            html.Div(col, className="sim-slider-header"),
                            dcc.Slider(
                                min=minimum, max=maximum, value=mean, marks=None,
                                tooltip=dict(placement='bottom', always_visible=True)
                            ),
                        ],
                        className="sim-slider"
                    )

                else:
                    uniques = stats_categoric[col]
                    options = [dict(label=f"{name}:{val}", value=name) for name,val in uniques.items()]
                    value   = uniques[0]['label']

                    field = dbc.InputGroup(
                        [
                            dbc.InputGroupText(col),
                            # Not a callback because it is the first inputable object.
                            dbc.Select(
                                options = options,
                                value   = value
                            ),
                        ],
                        size="sm", className='ctrl_chart ctrl_big ctr'
                    )
                kids.append(field)
    return kids
