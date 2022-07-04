# Local modules
from aiqc.orm import Dataset, Predictor
from aiqc import mlops
# UI modules
from dash import register_page, html, dcc, callback, ALL
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from uuid import uuid1
from scipy.special import logit


register_page(__name__)

refresh_seconds = 10*1000


# Dash naturally wraps in `id=_pages_content`
layout = dbc.Row(
    [
        dcc.Interval(
            id            = "initial_load",
            n_intervals   = 0,
            max_intervals = -1, 
            interval      = refresh_seconds
        ),
        dbc.Col(
            [
                html.H4('Scenario', className='sim-title'),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Model ID"),
                        # Not a callback because it is the first inputable object.
                        dbc.Select(id='pred_dropdown'),
                    ],
                    size="sm", className='ctrl_chart ctrl_big ctr'
                ),
                html.Br(),
                html.Div(id='scenario-btm'),
            ],
            width='3', align='center', className='sim-inputs'
        ),
        dbc.Col(
            [
                html.H4('Predictions', className='sim-title'),
                html.Div(id='sim-preds'),
            ],
            width='9', align='center', className='sim-outputs'
        ),
    ],
    className='sim-pane'
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
    ],
    Input(component_id="initial_load",  component_property="n_intervals"),
    State(component_id='pred_dropdown', component_property='value'),
)
def refresh_predictors(n_intervals:int, model_id:int):
    models = list(Predictor)
    if (not models):
        return [], "None yet"
    
    models.reverse()
    model_options = []
    for m in models:
        label = f"Model:{m.id}"
        if (m.is_starred):
            label += " ★"
        opt = dict(label=label, value=m.id)
        model_options.append(opt)
    
    return model_options, "Select model"


@callback(
    Output(component_id='scenario-btm', component_property='children'),
    Input(component_id='pred_dropdown', component_property='value')
)
def populate_features(model_id:int):
    if (model_id is None):
        msg   = "Select Model above"
        alert = [html.Br(), dbc.Alert(msg, className='alert')]
        return alert
    
    model = Predictor.get_by_id(model_id)
    features = model.job.queue.splitset.features

    f_kids = []
    for f in features:
        typ = f.dataset.typ
        # Need to check the type, so don't capitalize yet.
        subtitle = f"Features: {typ.capitalize()}"
        subtitle = html.P(subtitle, className='sim-subtitle')
        f_kids.append(subtitle)

        if (typ=='tabular'):
            stats_numeric   = f.dataset.stats_numeric
            stats_categoric = f.dataset.stats_categoric

            f_typs = f.get_dtypes()
            for col, typ in f_typs.items():
                is_numeric = np.issubdtype(typ, np.number)
                is_date    = np.issubdtype(typ, np.datetime64)
                
                if (is_numeric or is_date):
                    head = [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Stat"),
                                    html.Th("Value")
                                ]
                            )
                        ),
                    ]
                    
                    stats = stats_numeric[col]
                    body = []
                    for name,val in stats.items():
                        tip = html.Tr([
                            html.Td(name),
                            html.Td(f"{val:.3f}")
                        ])
                        body.append(tip)
                    body = [html.Tbody(body)]
                    # Don't use comma separated list
                    tips = dbc.Table(head + body)
                    
                    uid  = str(uuid1())
                    field = html.Div(
                        [
                            html.Div(col, id=uid, className='sim-slider-name'),
                            dbc.Tooltip(tips, target=uid, placement='right', className='sim-tooltip'),
                            dbc.Input(
                                id          = {'role':'feature', 'column':col},
                                type        = 'number',
                                value       = stats['50%'],
                                placeholder = stats['50%'],
                                className   = 'sim-slider-num'
                                # can't figure out validation tied to `step`
                            )
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
                f_kids.append(field)
    
    kids = [
        # Tried to put submit button below fields, but div heights tricky
        html.Div(
            dbc.InputGroup(
                [
                    dbc.Button(
                        "Simulate", outline=True, 
                        n_clicks=0, id="sim_button",
                        className='chart_button ctr',
                    ),
                ],
                size="md", className='ctrl_chart ctr'
            ), 
            className='sim-btn-block'
        ),
        html.Div(f_kids, className='sim-features')
    ]
    return kids

"""
Need both the value and id column name to construct df. 
id is created before value, and value is dynamic.
"""
@callback(
    Output('sim-preds', 'children'),
    Input('sim_button', 'n_clicks'),
    [
        State({'role': 'feature', 'column': ALL}, 'value'),
        State({'role': 'feature', 'column': ALL}, 'id'),
        State('pred_dropdown', 'value'),
        State('sim-preds', 'children')
    ]
)
def prediction_from_features(
    n_clicks:int
    , field_values:list
    , field_ids:list
    , model_id:int
    , preds:list
):
    if (n_clicks==0): raise PreventUpdate
    ### State needs the feature.id for accessing dataset

    # Construct records from feature fields
    record = {}
    for e, val in enumerate(field_values):
        col         = field_ids[e]['column']
        record[col] = val
    # All scalar values requires index
    df = pd.DataFrame.from_records(record, index=[0])

    model   = Predictor.get_by_id(model_id)
    feature = model.job.queue.splitset.features[0]
    f_cols  = feature.columns
    f_typs  = feature.get_dtypes()

    # Reorder the columns to match the original
    df = df.filter(items=f_cols)

    # Retype the columns using the original dtypes
    new_dset = Dataset.Tabular.from_df(dataframe=df, retype=f_typs)

    # Generate the prediction using high-level api
    prediction = mlops.Inference(
        predictor      = model,
        input_datasets = [new_dset]
    )
    
    # Information for the card body
    queue = prediction.predictor.job.queue
    label = queue.splitset.label
    if (label is not None):
        label = label.columns
        if (len(label)==1): label=label[0]
        label
    else:
        label = ""

    # Access the array, then the first prediction
    sim_val   = list(prediction.predictions.values())[0][0]
    sim_txt   = f"{label} = {sim_val}"
    sim_val   = html.Span(sim_txt, className='sim-val')
    pred      = html.P([f"Prediction #{prediction.id}: ", sim_val], className="card-head")
    pred      = dbc.Col(pred, width=9)

    mod_id    = html.Span("Model ID: ", className='card-subhead')
    mod_id    = html.P([mod_id, f"{model_id}"], className="card-text")
    mod_id    = dbc.Col(mod_id, width=3) 

    card_row  = dbc.Row([pred,mod_id], className='card-row')
    card_body = [card_row]

    analysis_typ = queue.algorithm.analysis_type
    if ('classification' in analysis_typ):
        fig = prediction.plot_confidence(call_display=False)
        fig = dcc.Graph(figure=fig, className='card-chart')
        card_body.append(fig)

    # Table of raw features for card footer
    cols  = [html.Th(col, className='sim-thead-th') for col in list(record.keys())]
    vals  = [html.Td(val, className='sim-td') for val in list(record.values())]
    head  = [html.Thead(html.Tr(cols, className='sim-thead-tr'), className='sim-thead')]
    body  = [html.Tbody(html.Tr(vals))]
    f_tbl = html.Table(head+body)
    f_tbl = html.Div(f_tbl, className='card-tbl-scroll')

    card = dbc.Card(
        [
            # dbc.CardHeader(pred_id),
            dbc.CardBody(card_body, className="card-bod"),  
            dbc.CardFooter(f_tbl, className="card-fut"),
        ],
        className="sim-card"
    )
    if (preds is None): preds=[]
    preds.insert(0,card)
    return preds
