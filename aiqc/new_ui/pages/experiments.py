# Local modules
from aiqc import orm
from aiqc.utils.meter import metrics_classify, metrics_regress
# UI modules
from dash import register_page, html, dcc, callback
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# Tried `redirect_from=['/']` but errors when root dir used
register_page(__name__, path='/')

refresh_seconds = 10*1000


# Dash naturally wraps in `id=_pages_content`
layout = html.Div(
    [
        dcc.Interval(id="initial_load", n_intervals=0, max_intervals=-1, interval=refresh_seconds),
        # ====== EXPERIMENT ======
        dbc.Row(
            [
                # =CONTROLS=
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Experiment ID"),
                            # Not a callback because it is the first inputable object.
                            dbc.Select(id='exp_dropdown'),
                        ],
                        size="sm", className='ctrl_chart ctrl_big ctr'
                    ),
                    width=3, align="center", className='ctrl_col_big'
                ),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Score Type"),
                            # Defined by downstream callback
                            dbc.Select(id='exp_type'),
                        ],
                        size="sm", className='ctrl_chart ctrl_big ctr'
                    ),
                    width=3, align="center", className='ctrl_col_big'
                ),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Min Score"),
                            dbc.Input(
                                type="number",id='exp_score',
                                placeholder="-∞"
                            ),
                        ],
                        size="sm", className='ctrl_chart ctrl_sm ctr'
                    ),
                    width=2, align="center", className='ctrl_col_sm'
                ),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Max Loss"),
                            dbc.Input(
                                type="number",id='exp_loss',
                                placeholder="+∞"
                            ),
                        ],
                        size="sm", className='ctrl_chart ctrl_sm ctr'
                    ),
                    width=2, align="center", className='ctrl_col_sm'
                ),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button(
                                "Filter", outline=True, 
                                n_clicks=0, id="exp_button",
                                className='chart_button ctr',
                            ),
                        ],
                        size="sm", className='ctrl_chart ctr'
                    ),
                    width=1, align="center",
                ),
            ],
            className='exp_ctrl_row'
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        dbc.Spinner(
                            color="info", delay_hide=1,
                            spinner_style={"width": "4rem", "height": "4rem"}
                        ),
                        className='spinner'
                    ),
                    id="exp_plot_contain", width=8, align="center"
                ),
                dbc.Col(
                    dbc.Alert(
                        "Click on a model in the chart on the left to show its parameters.",
                        className='alert'
                    ),
                    id="param_pane", className='hp_contain',
                    width=4, align="center"
                )
            ],
            className='exp_row'
        ),
        # ====== BOTTOM BAR ======
        # Stacked so sticks to top regardless of progress bar presence 
        html.Div(
            [
                # =PROGRESS BAR=
                # w/o gutter class the rounded edges of bar don't fill.
                dbc.Row(id="prog_container", className='g-0'),
                # =OPTIONS=
            ],
            className='prog_footer'
        ),
    ],
    className='page',
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
        Output(component_id="exp_dropdown", component_property="options"),
        Output(component_id="exp_dropdown", component_property="placeholder"),
        Output(component_id="exp_dropdown", component_property="value"),        
    ],
    Input(component_id="initial_load", component_property="n_intervals"),
    [
        State(component_id="exp_dropdown", component_property="value"),        
    ]
)
def refresh_experiments(n_intervals:int, queue_id:int):
    queues = list(orm.Queue)
    if (not queues):
        return [], "None yet", None
    
    # Initially the IDs are None becuse there is no State.
    queues.reverse()
    queue_options = [dict(label=str(q.id), value=q.id) for q in queues]
    if (queue_id is None):
        queue_id = queues[0].id

    return queue_options, None, queue_id


@callback(
    Output(component_id="prog_container", component_property="children"),
    Input(component_id="exp_dropdown",        component_property="value"),
)
def update_progress(queue_id:int):
    if (queue_id is None):
        return None
    queue = orm.Queue.get_by_id(queue_id)
    progress = round(queue.runs_completed/queue.total_runs*100)
    
    if (progress<100):
        children = dbc.Progress(
            id="progress_bar", className='prog_bar ctr', color="secondary",
            value=progress, label=f"{progress}%"
        )
        return children 
    else:
        return None


@callback(
    [
        Output(component_id='exp_type', component_property='options'),
        Output(component_id='exp_type', component_property='value'),
    ],
    Input(component_id='exp_dropdown', component_property='value'),
    [
        State(component_id='exp_type', component_property='value'),
    ]
)
def type_dropdown(queue_id:object, exp_type:str):
    if (queue_id is None):
        raise PreventUpdate
    
    queue = orm.Queue.get_by_id(queue_id)
    analysis_type = queue.algorithm.analysis_type
    if ('classification' in analysis_type):
        score_types = metrics_classify
        if (exp_type not in score_types):
            exp_type = 'accuracy'
    elif('regression' in analysis_type):   
        score_types = metrics_regress
        if (exp_type not in score_types):
            exp_type = 'r2'
    
    options = [{"label":m, "value":col} for col, m in score_types.items()]
    return options, exp_type


@callback(
    Output(component_id='exp_plot_contain', component_property='children'),
    Input(component_id='exp_button',        component_property='n_clicks'),
    [
        State(component_id='exp_dropdown', component_property='value'),
        State(component_id='exp_type',     component_property='value'),
        State(component_id='exp_score',    component_property='value'),
        State(component_id='exp_loss',     component_property='value'),
    ]
)
def plot_experiment(
    n_clicks:int, queue_id:object,
    score_type:str, min_score:float, max_loss:float,
):   
    if (queue_id is None):
        queues = list(orm.Queue)
        if (not queues):
            msg = "Sorry - Cannot display plot because no Queues exist yet. Data will refresh automatically."
            return dbc.Alert(msg, className='alert')
        else:
            # Plot the most recent by default
            queue = queues[-1]
    else:
        queue = orm.Queue.get_by_id(queue_id)
    
    try:
        fig = queue.plot_performance(
            score_type=score_type, min_score=min_score, max_loss=max_loss,
            call_display=False, height=500
        )
    except Exception as err_msg:
        return dbc.Alert(str(err_msg), className='alert')
    else:
        fig = dcc.Graph(id="exp_plot", figure=fig, className='exp_plot_contain ctr')
        return fig


@callback(
    Output(component_id='exp_button', component_property='n_clicks'),
    # If the type is changed, don't wait for button to be clicked.
    Input(component_id='exp_type',    component_property='value'),
    State(component_id='exp_button',  component_property='n_clicks'),
)
def trigger_graph(score_type:str, n_clicks:int):
    n_clicks += 1
    return n_clicks


@callback(
    Output(component_id='param_pane', component_property='children'),
    # `Input...clickAnnotationData` returns None
    Input(component_id='exp_plot',    component_property='clickData'),
)
def interactive_params(new_click:dict):
    """
    It's hard to maintain clickData State because exp_plot doesn't exist when 
    the page loads and exp_plot gets overrode when the page refreshes.
    """
    if (new_click is None):
        raise PreventUpdate
    # The docs say to use `json.dumps`, but clickData is just a dict.
    predictor   = new_click['points'][0]['customdata'][0]    
    predictor   = orm.Predictor.get_by_id(predictor)
    title       = f"Model ID: {predictor}"
    model_title = html.P(title, className='header')
    hp_table    = fetch_params(predictor, "tbig")
    return [model_title, hp_table]
