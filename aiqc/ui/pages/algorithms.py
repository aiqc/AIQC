# Local modules
from aiqc import orm
# UI modules
from dash import register_page, html, dcc, callback
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc

register_page(__name__)

# Dash naturally wraps in `id=_pages_content`
layout = html.Div(
    [
        dcc.Interval(
            id              = "initial_load"
            , n_intervals   = 0
            , max_intervals = -1
            , interval      = 10*1000
        ),
        dbc.Row(
            [
                dbc.Col(width="4"),
                dbc.Col(
                    dcc.Dropdown(id="multitron", multi=True, className='multitron ctr'),
                    width="4", align="center",
                ),
                dbc.Col(width="3"),
                dbc.Col(
                    html.Div(html.A("⇪",href="#", className='up_arrow')),
                    width="1", align="center",
                )
            ],
            className='middle_bar'
        ),
        html.Br(),html.Br(),html.Br(),
        dbc.Row(id="model_container", className='model_container'),
        html.Br(),
    ]
)


# Helper functions for callbacks.
def fetch_params(predictor:object, size:str):
    hyperparameters = predictor.get_hyperparameters()
    if (hyperparameters is not None):
        headers = [html.Th("parameter"), html.Th("value")]
        table_header = [html.Thead(html.Tr(headers), className='thead')]
        # bools are not rendering so need to force them to str
        rows = []
        for k,v in hyperparameters.items():
            if isinstance(v,bool):
                v = str(v) 
            rows.append(
                html.Tr([html.Td(k), html.Td(v)])
            )
        table_body = [html.Tbody(rows)]
        hp_table = dbc.Table(
            table_header + table_body,
            dark=True, hover=True, responsive=True,
            striped=True, bordered=False, className=f"tbl {size} ctr"
        )
    else:
        msg = "Sorry - This model has no parameters."
        hp_table = dbc.Alert(msg, className='alert')
    return hp_table


"""
- When the page is started and refreshed, this updates things.
- You can only use objects at the start of the DAG, 
  e.g. not `exp_plot.clickData` because it doesn't exist yet.
"""
@callback(
    [       
        Output(component_id='multitron',    component_property='options'),
        Output(component_id='multitron',    component_property='placeholder'),
        Output(component_id="multitron",    component_property="value"),
    ],
    Input(component_id="initial_load", component_property="n_intervals"),
    [
        State(component_id='multitron',    component_property='value'),
    ]
)
def refresh_models(n_intervals:int, model_ids:int):
    models = list(orm.Predictor)
    if (not models):
        return [], "None yet", []
    models.reverse()
    model_options = [dict(label=f"Model:{m.id}", value=m.id) for m in models]
    
    return model_options, "Compare models head-to-head", model_ids


@callback(
    Output(component_id='model_container', component_property='children'),
    Input(component_id='multitron',        component_property='value')
)
def model_plots(predictor_ids:list):
    # Initially it's None, but empty list when it's cleared.
    if ((predictor_ids is None) or (not predictor_ids)):
        msg = "Select up to two models from the dropdown above."
        return dbc.Alert(msg, className='alert', style={"width":"50%"})
    pred_count = len(predictor_ids)
    if (pred_count==1):
        col_width = 12
    elif (pred_count==2):
        col_width = 6
    elif (pred_count>2):
        msg = "Sorry - Only 2 models can be displayed at once."
        return dbc.Alert(msg, className='alert')
    multi_cols = []
    for predictor_id in predictor_ids:
        # Only `big_column` is assigned a bootstrap width.
        big_column = []

        predictor = orm.Predictor.get_by_id(predictor_id)
        predictions = list(predictor.predictions)
        if (not predictions):
            msg = f"Sorry - Metrics for this model are not ready yet. Data will refresh automatically."
            return dbc.Col(dbc.Alert(msg, className='alert'))
        prediction = predictions[0]

        # === METRICS ===
        metrics = prediction.metrics
        # Need the 'split' to be in the same dict as the metrics.
        metrics_records = []
        for split, metrix in metrics.items():
            if (metrix is not None):
                split_dikt = {'split':split}
                # We want the split to appear first in the dict
                split_dikt.update(metrix)
                metrics_records.append(split_dikt)
        cols         = list(metrics_records[0].keys())
        headers      = [html.Th(c) for c in cols]
        table_header = [html.Thead(html.Tr(headers))]

        metrics_raw = []
        for record in metrics_records:
            metrics = [v for k,v in record.items()]
            metrics_raw.append(metrics)

        rows = []
        for cells in metrics_raw:
            row = html.Tr([html.Td(cell) for cell in cells]) 
            rows.append(row)            
        table_body = [html.Tbody(rows)]
        metrics_table = dbc.Table(
            table_header + table_body
            , dark       = True
            , hover      = True
            , responsive = True
            , striped    = True
            , bordered   = False
            , className  = 'tbl tbig ctr'
        )

        row = dbc.Row(
            dbc.Col(metrics_table, align="center")
        )
        big_column.append(row)
        big_column.append(html.Hr(className='hrz ctr'))

        # === HYPERPARAMETERS ===
        hp_table = fetch_params(predictor, "tsmall")
        row = dbc.Row(
            dbc.Col(hp_table, align="center")
        )
        big_column.append(row)
        big_column.append(html.Hr(className='hrz ctr'))

        # === LEARNING ===
        learning_curves = predictor.plot_learning_curve(
            call_display=False, skip_head=False
        )
        learning_curves = [dcc.Graph(figure=fig, className='plots ctr') for fig in learning_curves]
        row = dbc.Row(
            [
                dbc.Col(learning_curves, id="learning_plots"),
            ]
        )
        big_column.append(row)
        big_column.append(html.Hr(className='hrz ctr'))

        # === IMPORTANCE ===
        feature_importance = prediction.feature_importance
        if (feature_importance is not None):
            prediction = orm.Predictor.get_by_id(predictor_id).predictions[0]
            content    = prediction.plot_feature_importance(top_n=15, call_display=False)
            content    = [dcc.Graph(figure=fig, className='plots ctr') for fig in content]
        elif (feature_importance is None):
            msg = "Feature importance not calculated for model yet."
            content = dbc.Alert(msg, className='alert')

        row = dbc.Row(
            dbc.Col(content, id="importance_plots")
        )
        big_column.append(row)
        big_column.append(html.Hr(className='hrz ctr'))

        # === CLASSIFICATION ===
        analysis_type = predictor.job.queue.algorithm.analysis_type
        if ('classification' in analysis_type):
            # === ROC ===
            roc = prediction.plot_roc_curve(call_display=False)
            roc = dcc.Graph(figure=roc, className='plots ctr')
            row = dbc.Row(
                dbc.Col(roc, align="center")
            )
            big_column.append(row)
            big_column.append(html.Hr(className='hrz ctr'))
            # === PRC ===
            pr = prediction.plot_precision_recall(call_display=False)
            pr = dcc.Graph(figure=pr, className='plots ctr')
            row = dbc.Row(
                dbc.Col(pr, align="center")
            )
            big_column.append(row)
            big_column.append(html.Hr(className='hrz ctr'))
            # === CONFUSION ===
            cms = prediction.plot_confusion_matrix(call_display=False)
            cms = [dcc.Graph(figure=fig, className='plots ctr') for fig in cms]                
            row = dbc.Row(
                dbc.Col(cms, align="center")
            )
            big_column.append(row)
            big_column.append(html.Br())
            big_column.append(html.Br())
        big_column = dbc.Col(big_column, width=col_width)
        multi_cols.append(big_column)
    return multi_cols