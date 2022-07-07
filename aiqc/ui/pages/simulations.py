# Local modules
from aiqc.orm import Dataset, Predictor, Prediction
from aiqc import mlops
# UI modules
from dash import register_page, html, dcc, callback, ALL, MATCH
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
# Wrangling modules
import numpy as np
import pandas as pd
from uuid import uuid1

register_page(__name__)

# Dash naturally wraps in `id=_pages_content`
layout = dbc.Row(
    [
        dcc.Interval(
            id              = "initial_load"
            , n_intervals   = 0
            , max_intervals = -1
            , interval      = 10*1000
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
                html.Div(
                    [
                        html.Br(),
                        dbc.Alert(
                            "No predictions have been made yet. To begin, select a model on the left.",
                            className='alert'
                        ),
                    ],
                    id='sim-preds'
                ),
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
        # Does not overwrite the value, just updates the options
    ],
    Input(component_id="initial_load",  component_property="n_intervals"),
)
def refresh_predictors(n_intervals:int):
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
    if (model_id is None): raise PreventUpdate

    model = Predictor.get_by_id(model_id)
    splitset = model.job.queue.splitset

    if (splitset.supervision=='unsupervised'):
        return [
            html.Br(),
            dbc.Alert(
                "Unsupervised analysis not supported in UI yet.",
                className='alert'
            )
        ]

    features = splitset.features
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

            # Ideally we want to show features in rank order of importance
            importance = model.predictions[0].feature_importance
            f_typs     = f.get_dtypes()
            if (importance is not None):
                imp_df       = model.predictions[0].importance_df(feature_id=f.id)
                ranked_feats = list(imp_df['Feature'])
            else:
                ranked_feats = list(f_typs.keys())
            
            for col in ranked_feats:
                typ        = f_typs[col]
                is_numeric = np.issubdtype(typ, np.number)
                is_date    = np.issubdtype(typ, np.datetime64)
                
                # Assemble the feature metadata tooltip
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
                    tbod = []
                    # Feature importance metadata
                    if (importance is not None):
                        imp = float(imp_df[imp_df['Feature']==col]['Median'])
                        tip = html.Tr([
                            html.Td('importance'),
                            html.Td(f"{imp:.3f}")
                        ])
                        tbod.append(tip)
                    # Feature distribution metadata
                    stats = stats_numeric[col]
                    for name,val in stats.items():
                        tip = html.Tr([
                            html.Td(name),
                            html.Td(f"{val:.3f}")
                        ])
                        tbod.append(tip)
                    tbod = [html.Tbody(tbod)]
                    # Don't use comma separated list
                    tips = dbc.Table(head + tbod)
                    
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
dash.plotly.com/pattern-matching-callbacks
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
    # Remember, n_clicks resets when changing model dropdown
    if (n_clicks==0): 
        raise PreventUpdate
    elif (n_clicks==1): 
        preds = []
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

    analysis_typ = queue.algorithm.analysis_type
    # Access the array, then the first prediction
    sim_val   = list(prediction.predictions.values())[0][0]
    if ('regression' in analysis_typ):    
        sim_txt = f"{label} = {sim_val:.3f}"
    else:
        sim_txt = f"{label} = {sim_val:}"

    pred_id = prediction.id
    starred = prediction.is_starred
    if (starred==False):
        star = '☆'
    else:
        star = '★'
    star = dbc.Button(star,id={'role':'pred_star','pred_id':pred_id}, className='pred_star', color='link')###

    sim_val   = html.Span(sim_txt, className='sim-val')
    pred      = html.P([star, f"Prediction #{prediction.id}: ", sim_val], className="card-head")
    pred      = dbc.Col(pred, width=9)

    mod_id    = html.Span("Model ID: ", className='card-subhead')
    mod_id    = html.P([mod_id, f"{model_id}"], className="card-text")
    mod_id    = dbc.Col(mod_id, width=3) 

    card_row  = dbc.Row([pred,mod_id], className='card-row')
    card_body = [card_row]

    if ('classification' in analysis_typ):
        fig = prediction.plot_confidence(call_display=False)
        fig = dcc.Graph(figure=fig, className='card-chart')
        card_body.append(fig)

    # Table of raw features for card footer
    cols  = list(record.keys())
    cols  = [html.Th(col, className='sim-thead-th') for col in cols]
    vals  = list(record.values())
    vals  = [round(val,3) if isinstance(val,float) else val for val in vals]
    vals  = [html.Td(val, className='sim-td') for val in vals]
    head  = [html.Thead(html.Tr(cols, className='sim-thead-tr'), className='sim-thead')]
    body  = [html.Tbody(html.Tr(vals))]
    f_tbl = html.Table(head+body)
    f_tbl = html.Div(
        html.Div(
            f_tbl, className='card-tbl'
        ),
        className='card-tbl-scroll'
    )

    card = dbc.Card(
        [
            dbc.CardBody(card_body, className="card-bod"),  
            dbc.CardFooter(f_tbl, className="card-fut"),
        ],
        className="sim-card"
    )
    preds.insert(0,card)
    return preds


@callback(
    Output({'role':'pred_star', 'pred_id': MATCH}, 'children'),
    Input({'role':'pred_star', 'pred_id': MATCH}, 'n_clicks'),
    [
        State({'role':'pred_star', 'pred_id': MATCH}, 'children'),
        State({'role':'pred_star', 'pred_id': MATCH}, 'id'),
    ]
)
def flip_pred_star(n_clicks, star, id):
    if (n_clicks is None):
        raise PreventUpdate
    if (star=='☆'):
        star = '★'
    else:
        star = '☆'
    Prediction.get_by_id(id['pred_id']).flip_star()
    return star
