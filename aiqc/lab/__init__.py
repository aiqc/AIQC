# Local modules.
from .. import orm
from ..utils import metrics_classify, metrics_regress
# External modules.
from logging import getLogger, ERROR
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from jupyter_dash import JupyterDash

def launch(server_runtime:dict=None):
    """`server_runtime` is passed through to `dash.Dash.app.run_server()` as **kwargs."""
    # css uses `-` which aren't compatible with `dict(key)` syntax
    # merge two dictionaries: `z = {**x, **y}`
    style = dict(###
        body = {
            "background":"#182d41",
            "z-index":"111",
        },
        # ====== NAV ======
        nav = {
            "background-image":"linear-gradient(#1b344a,rgb(1, 10, 17))",
            "height": "48px",
            # These 3 stick it to top.
            "position": "sticky",
            "top": "0",
            "z-index": "999",
        },
        logo_aligner = {
            "position": "relative",
            "top": "44%",
            "-webkit-transform": "translateY(-44%)",
            "-ms-transform": "translateY(-44%)",
            "transform": "translateY(-44%)",
        },
        nav_link = {
            "textDecoration":"none",
            "margin-left":"12px",
        },
        # ====== MIDDLE BAR ======
        middle_bar = {
            "background-image": "linear-gradient(rgb(1, 10, 17),rgba(1, 10, 17, 0.7))",
            # These 3 stick it to top.
            "position": "sticky",
            "top": "48px",#same as nav height.
            "z-index": "999",#same as nav index.
            "padding-top": "18px",
            "padding-bottom": "18px",
            "padding-left":"3%",
            "padding-right":"3%",
        },
        # ====== PROGRESS ======
        progress = {
            "width": "45%",
            "margin-right": "auto",
            "margin-bottom": "30px",
            "margin-left": "auto",
            "border": "1px solid #6c757d",#matches secondary color
            "border-radius": "12px",
            "background": "none",
            "height": "15px",
            "font-size": "12px",
            "color": "rgb(1, 10, 17)",
        },
        # ====== MODEL OPTIONS ======
        up_arrow = {
            "color": "rgb(94, 118, 140)",
            "text-decoration": "none",
            "font-size": "19px",
        },
        # ====== CHART BUTTONS ======
        chart_ctrl = {
            "width":"75%",
            "margin-left":"auto",
            "margin-right":"auto",
        },
        chart_ctrl_big = {
            "min-width": "212px",
            "width":"75%",
            "margin-left":"auto",
            "margin-right":"auto",
        },
        chart_ctrl_sm = {
            "min-width": "160px",
            "width":"75%",
            "margin-left":"auto",
            "margin-right":"auto",
        },
        chart_button = {
            "margin-left": "auto",
            "margin-right": "auto",
            "padding-top": "7px",
            "padding-bottom": "7px",
            "padding-left": "12px",
            "padding-right": "12px",
            "border-radius": "6px",
            "color": "#43eac5",
            "border-color": "#43eac5",
            "font-weight": "lighter",
        },
        ctrl_col_big={"min-width":"230px"},
        ctrl_col_sm={"min-width":"190px"},
        multitron = {
            "margin-left": "auto",
            "margin-right": "auto",
        },
        # ====== TABLES ======
        hp_contain = {
            "overflow-y": "scroll",
            "max-height": "100%",
            "margin-top": "20px",
            "padding-right":"5%",
        },
        tables = {
            "text-align": "center", 
            "font-size": "14px",
            "margin-left": "auto",
            "margin-right": "auto",
            "margin-bottom": "5px",#it injects its own w/o this
            # border-radius not being applied
        },
        tbig = {"width":"70%"},
        tsmall = {"width":"26%"},
        thead = {"font-size":"14.5px","color":"white"}, #bootstrap overrides thead styles.
        tbody = {"border-top-color": "#484848"}, # bootstrap overrides tbody styles.
        # ====== PLOTS ======
        exp_plot_contain = {
            "width": "90%",
            "margin-left": "auto",
            "margin-right": "auto",
            "height":"500px",
        },
        plots = {
            "width": "90%",
            "margin-left": "auto",
            "margin-right": "auto",
        },
        # ====== FORMAT ======
        hr = {
            "color": "#40566b",
            "width": "85%",
            "height": "3px",
            "margin-top":"45px",
            "margin-bottom":"45px",
            "margin-left":"auto",
            "margin-right":"auto",
        },
        header = {
            "color": "#ffda67",
            "font-size": "18px",
            "font-weight": "200",
            "font-stretch": "108%",
            "text-align": "center",
        },
        exp_ctrl_row = {
            "margin-left": "3%",
            "margin-top":"15px",
        },
        exp_row = {
            "margin-left": "3%",
            "height": "500px",
            "margin-bottom": "15px",
        },
        alert = {
            "width": "80%",
            "margin": "auto",
            "text-align": "center",
            "background": "#ffffff20",
            "border-radius": "12px",
            "border-color": "#ffffff00",
            "border-width": "4px",
            "padding": "25px",
            "color": "white",
            "font-weight": "lighter",
        },
        # ====== MODEL PLOTS ======
        spinner = {"margin-left": "48%"},
    )


    # remote cdn: https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css
    sheets = [dbc.themes.BOOTSTRAP]
    ###app = dash.Dash(__name__)
    app = JupyterDash(__name__, external_stylesheets=sheets)
    # When using components that are generated by other callbacks
    app.config['suppress_callback_exceptions'] = True
    ### server = app.server

    # Globals for app
    logo_path = "https://raw.githubusercontent.com/aiqc/AIQC/main/docs/images/logo_wide_small.svg"


    app.layout = html.Div(
        [
            dcc.Interval(id="initial_load", n_intervals=0, max_intervals=-1, interval=4000),
            # ====== NAVBAR ======
            html.Div(
                html.Center(
                    html.A(
                        html.Img(src=logo_path, height="35px"),
                        href="https://aiqc.readthedocs.io",
                        style=style['nav_link'],
                    ),
                    style=style['logo_aligner'],
                ),
                style=style['nav']
            ),
            # Grid system Div(Row(Col(Div)))
            # dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
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
                            size="sm", style=style['chart_ctrl_big']
                        ),
                        width=3, align="center", style=style['ctrl_col_big']
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Score Type"),
                                # Defined by downstream callback
                                dbc.Select(id='exp_type'),
                            ],
                            size="sm", style=style['chart_ctrl_big']
                        ),
                        width=3, align="center", style=style['ctrl_col_big']
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
                            size="sm", style=style['chart_ctrl_sm']
                        ),
                        width=2, align="center", style=style['ctrl_col_sm']
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
                            size="sm", style=style['chart_ctrl_sm']
                        ),
                        width=2, align="center", style=style['ctrl_col_sm']
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.Button(
                                    "Filter", outline=True, 
                                    n_clicks=0, id="exp_button",
                                    style=style['chart_button'],
                                ),
                            ],
                            size="sm", style=style['chart_ctrl']
                        ),
                        width=1, align="center",
                    ),
                ],
                style=style['exp_ctrl_row'],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dbc.Spinner(
                                color="info", delay_hide=1,
                                spinner_style={"width": "4rem", "height": "4rem"}
                            ),
                            style=style['spinner']
                        ),
                        id="exp_plot_contain", width=8, align="center"
                    ),
                    dbc.Col(
                        dbc.Alert(
                            "Click on a model in the chart on the left to show its parameters.",
                            style=style['alert']
                        ),
                        id="param_pane", style=style['hp_contain'],
                        width=4, align="center",
                    )
                ],
                style=style['exp_row'],
            ),
            # ====== MIDDLE BAR ======
            # Stacked so sticks to top regardless of progress bar presence 
            html.Div(
                [
                    # =PROGRESS BAR=
                    # w/o `g-0` gutter class the rounded edges of bar don't fill.
                    dbc.Row(id="progress_container", className="g-0"),
                    # =OPTIONS=
                    dbc.Row(
                        [
                            dbc.Col(width="4"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="multitron", multi=True, style=style['multitron']
                                ),
                                width="4", align="center",
                            ),
                            dbc.Col(width="3"),
                            dbc.Col(
                                html.Div(html.A("⇪",href="#",style=style['up_arrow'])),
                                width="1", align="center",
                            )  
                        ]
                    )
                ],
                style=style['middle_bar'],
            ),
            html.Br(),html.Br(),html.Br(),
            dbc.Row(id="model_container"),
            html.Br(),html.Br(),html.Br(),
        ],
        style=style['body'],
    )


    # Helper functions for callbacks.
    def fetch_params(predictor:object):
        hyperparameters = predictor.get_hyperparameters()
        if (hyperparameters is not None):
            headers = [html.Th("parameter"), html.Th("value")]
            table_header = [html.Thead(html.Tr(headers), style=style['thead'])]
            rows = [html.Tr([html.Td(k), html.Td(v)]) for k,v in hyperparameters.items()]
            table_body = [html.Tbody(rows, style=style['tbody'])]
            hp_table = dbc.Table(
                table_header + table_body,
                dark=True, hover=True, responsive=True,
                striped=True, bordered=False, style={**style['tables'],**style['tsmall']}
            )
        else:
            msg = "Sorry - This model has no parameters."
            hp_table = dbc.Alert(msg, style=style['alert'])
        return hp_table


    """
    - When the page is started and refreshed, this updates things.
    - You can only use objects at the start of the DAG, 
    e.g. not `exp_plot.clickData` because it doesn't exist yet.
    """
    @app.callback(
        [
            Output(component_id="exp_dropdown", component_property="options"),
            Output(component_id="exp_dropdown", component_property="placeholder"),
            Output(component_id="exp_dropdown", component_property="value"),        
            Output(component_id='multitron', component_property='options'),
            Output(component_id='multitron', component_property='placeholder'),
            Output(component_id="multitron", component_property="value"),
        ],
        Input(component_id="initial_load", component_property="n_intervals"),
        [
            State(component_id="exp_dropdown", component_property="value"),        
            State(component_id='multitron', component_property='value'),
        ]
    )
    def refresh(n_intervals:int, queue_id:int, model_ids:int):
        queues = list(orm.Queue)
        if (not queues):
            return [], "None yet", None, [], "None yet", []
        
        # Initially the IDs are None becuse there is no State.
        queues.reverse()
        queue_options = [dict(label=str(q.id), value=q.id) for q in queues]
        if (queue_id is None):
            queue_id = queues[0].id
        
        models = list(orm.Predictor)
        models.reverse()
        model_options = [dict(label=f"Model:{m.id}", value=m.id) for m in models]
        
        return queue_options, None, queue_id, model_options, "Compare models head-to-head", model_ids


    @app.callback(
        Output(component_id="progress_container", component_property="children"),
        Input(component_id="exp_dropdown", component_property="value"),
    )
    def update_progress(queue_id:int):
        if (queue_id is None):
            return None
        queue = orm.Queue.get_by_id(queue_id)
        progress = round(queue.runs_completed/queue.run_count*100)
        
        if (progress<100):
            children = dbc.Progress(
                id="progress_bar", style=style['progress'], color="secondary",
                value=progress, label=f"{progress}%"
            )
            return children 
        else:
            return None


    @app.callback(
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


    @app.callback(
        Output(component_id='exp_plot_contain', component_property='children'),
        Input(component_id='exp_button', component_property='n_clicks'),
        [
            State(component_id='exp_dropdown', component_property='value'),
            State(component_id='exp_type', component_property='value'),
            State(component_id='exp_score', component_property='value'),
            State(component_id='exp_loss', component_property='value'),
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
                return dbc.Alert(msg, style=style['alert'])
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
            return dbc.Alert(str(err_msg), style=style['alert'])
        else:
            fig = dcc.Graph(id="exp_plot", figure=fig, style=style['exp_plot_contain'])
            return fig


    @app.callback(
        Output(component_id='exp_button', component_property='n_clicks'),
        Input(component_id='exp_type', component_property='value'),
        State(component_id='exp_button', component_property='n_clicks'),
    )
    def trigger_graph(score_type:str, n_clicks:int):
        n_clicks += 1
        return n_clicks


    @app.callback(
        Output(component_id='param_pane', component_property='children'),
        # `Input...clickAnnotationData` returns None
        Input(component_id='exp_plot', component_property='clickData'),
    )
    def interactive_params(new_click:dict):
        """
        It's hard to maintain clickData State because exp_plot doesn't exist when 
        the page loads and exp_plot gets overrode when the page refreshes.
        """
        if (new_click is None):
            raise PreventUpdate
        # The docs say to use `json.dumps`, but clickData is just a dict.
        predictor = new_click['points'][0]['customdata'][0]    
        new_predCache = dict(id=predictor)
        predictor = orm.Predictor.get_by_id(predictor)
        title = f"Model ID: {predictor}"
        model_title = html.P(title, style=style['header'])
        hp_table = fetch_params(predictor)
        
        return [model_title, html.Br(), hp_table]


    @app.callback(
        Output(component_id='model_container', component_property='children'),
        Input(component_id='multitron', component_property='value')
    )
    def model_plotz(predictor_ids:list):
        # Initially it is None as opposed to an empty list
        if (predictor_ids is None):
            raise PreventUpdate
        pred_count = len(predictor_ids)
        if (pred_count==1):
            col_width = 12
            conf_width = None
        elif (pred_count==2):
            col_width = 6
            conf_width = "auto"
        elif (pred_count>2):
            msg = "Sorry - Only 2 models can be displayed at once."
            return dbc.Alert(msg, style=style['alert'])
        multi_cols = []
        for predictor_id in predictor_ids:
            # Only `big_column` is assigned a bootstrap width.
            big_column = []

            predictor = orm.Predictor.get_by_id(predictor_id)
            predictions = list(predictor.predictions)
            if (not predictions):
                msg = f"Sorry - Metrics for this model are not ready yet. Data will refresh automatically."
                return dbc.Col(dbc.Alert(msg, style=style['alert']))
            prediction = predictions[0]

            # === METRICS ===
            metrics = prediction.metrics
            # Need the 'split' to be in the same dict as the metrics.
            metrics_records = []
            for split, metrix in metrics.items():
                split_dikt = {'split':split}
                # We want the split to appear first in the dict
                split_dikt.update(metrix)
                metrics_records.append(split_dikt)
            cols = list(metrics_records[0].keys())
            headers = [html.Th(c) for c in cols]
            table_header = [html.Thead(html.Tr(headers),style=style['thead'])]

            metrics_raw = []
            for record in metrics_records:
                metrics = [v for k,v in record.items()]
                metrics_raw.append(metrics)

            rows = []
            for cells in metrics_raw:
                row = html.Tr([html.Td(cell) for cell in cells]) 
                rows.append(row)            
            table_body = [html.Tbody(rows, style=style['tbody'])]
            metrics_table = dbc.Table(
                table_header + table_body,
                dark=True, hover=True, responsive=True, 
                striped=True, bordered=False, style={**style['tables'],**style['tbig']}
            )

            row = dbc.Row(
                [
                    dbc.Col(metrics_table, align="center",),
                ]
            )
            big_column.append(row)
            big_column.append(html.Hr(style=style['hr']))

            # === HYPERPARAMETERS ===
            hp_table = fetch_params(predictor)
            row = dbc.Row(
                [
                    dbc.Col(hp_table, align="center",),
                ]
            )
            big_column.append(row)
            big_column.append(html.Hr(style=style['hr']))


            # === LEARNING ===
            learning_curves = predictor.plot_learning_curve(
                call_display=False, skip_head=False
            )
            learning_curves = [dcc.Graph(figure=fig, style=style['plots']) for fig in learning_curves]
            row = dbc.Row(
                [
                    dbc.Col(learning_curves, id="learning_plots"),
                ]
            )
            big_column.append(row)
            big_column.append(html.Hr(style=style['hr']))

            # === IMPORTANCE ===
            feature_importance = prediction.feature_importance
            if (feature_importance is not None):
                prediction = orm.Predictor.get_by_id(predictor_id).predictions[0]
                content = prediction.plot_feature_importance(top_n=15, call_display=False)
                content = [dcc.Graph(figure=fig, style=style['plots']) for fig in content]
            elif (feature_importance is None):
                msg = "Feature importance not calculated for model yet."
                content = dbc.Alert(msg, style=style['alert'])

            row = dbc.Row(
                [
                    dbc.Col(content, id="importance_plots"),
                ]
            )
            big_column.append(row)
            big_column.append(html.Hr(style=style['hr']))

            # === CLASSIFICATION ===
            analysis_type = predictor.job.queue.algorithm.analysis_type
            if ('classification' in analysis_type):
                # === ROC ===
                roc = prediction.plot_roc_curve(call_display=False)
                roc = dcc.Graph(figure=roc, style=style['plots'])
                row = dbc.Row(
                    [
                        dbc.Col(roc, align="center",),
                    ]
                )
                big_column.append(row)
                big_column.append(html.Hr(style=style['hr']))
                # === PRC ===
                pr = prediction.plot_precision_recall(call_display=False)
                pr = dcc.Graph(figure=pr, style=style['plots'])
                row = dbc.Row(
                    [
                        dbc.Col(pr, align="center",),
                    ]
                )
                big_column.append(row)
                big_column.append(html.Hr(style=style['hr']))
                # === CONFUSION ===
                cms = prediction.plot_confusion_matrix(call_display=False, width=conf_width)
                cms = [dcc.Graph(figure=fig, style=style['plots']) for fig in cms]                
                row = dbc.Row(
                    [
                        dbc.Col(cms, align="center",),
                    ]
                )
                big_column.append(row)
                big_column.append(html.Br())
                big_column.append(html.Br())
            big_column = dbc.Col(big_column, width=col_width)
            multi_cols.append(big_column)
        return multi_cols


    log = getLogger('werkzeug')
    log.setLevel(ERROR)

    # Feels silly to define a user-facing class w __init__ for a single method.
    runtime = dict(
        mode = "external"
        , debug = False
        , host = '127.0.0.1' 
        , port = '9991'# just keep incrementing if it's taken.
    )
    if (server_runtime is not None):
        for parameter, value in server_runtime.items():
            runtime[parameter] = value
    app.run_server(**runtime)
