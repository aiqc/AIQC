"""
Plots
└── Documentation = https://aiqc.readthedocs.io/en/latest/notebooks/visualization.html
└── Data is prepared in the `plot_*` methods of the Queue and Predictor ORM classes.
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from re import sub


def set_dash_font(fig:object):
    """Dash knows about `/lab/assets` for Abel .tff, but regular charts don't"""
    return fig.update_layout(
        font         = dict(family="Abel, Avenir", size=14)
        , titlefont  = dict(family="Abel, Avenir")
        , hoverlabel = dict(font=dict(family="Abel, Avenir"))
    )


class Plot(object):
    def __init__(self):
        """
        - Defines a template that is passed to the other plots.
        - Font gets overriden to Abel below by Dash.
        """
        self.plot_template = dict(
            layout=go.Layout(
                font            = dict(family="Avenir", color='#FAFAFA', size=13)
                , title         = dict(x=0.05, y=1.0)
                , titlefont     = dict(family="Avenir")
                , title_pad     = dict(b=50, t=20)
                , plot_bgcolor  = '#182d41'
                , paper_bgcolor = '#182d41'
                , hovermode     = 'closest'
                , hoverlabel    = dict(
                    bgcolor = "#122536"
                    , font  = dict(family="Avenir", size=15)
                )
            )
        )


    def performance(
        self
        , dataframe:object
        , score_type:str
        , score_display:str
        , call_display:bool = True
        , height:int        = None
    ):
        fig = px.line(
            dataframe
            , title      = 'Model Metrics by Split'
            , x          = 'loss'
            , y          = score_type
            , color      = 'predictor_id'
            , height     = height
            , hover_data = ['predictor_id', 'split', 'loss', score_type]
            , line_shape = 'spline'
        )
        fig.update_traces(
            mode     = 'markers+lines'
            , line   = dict(width = 2)
            , marker = dict(
                size   = 8
                , line = dict(width=2, color='white')
            )
        )
        fig.update_layout(
            title_x       = 0.5#center
            , xaxis_title = "Loss"
            , yaxis_title = score_display
            , template    = self.plot_template
            , showlegend  = False#gets way too busy
        )

        fig.update_xaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        fig.update_yaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        
        if (call_display==True):
            fig.show()
        else:
            fig = set_dash_font(fig)
            return fig


    def learning_curve(
        self
        , dataframe:object
        , history_pairs:dict
        , skip_head:bool    = False
        , call_display:bool = True
    ):
        """Dataframe rows are epochs and columns are metric names."""
        if (skip_head==True):
            dataframe = dataframe.tail(round(dataframe.shape[0]*.85))
        # Spline seems to crash with too many points.
        if (dataframe.shape[0] >= 400):
            line_shape = 'linear'
        elif (dataframe.shape[0] < 400):
            line_shape = 'spline'

        # Create a plot for every pair.
        figs = []
        for train,val in history_pairs.items():
            metric_name = sub("_"," ",train[6:])
            df = dataframe[[train,val]]
            # The y axis is so variable that we need a range for it
            vals_all = list(df[train]) + list(df[val])
            tick_size = abs((max(vals_all) - min(vals_all))/7)
            fig = px.line(
                df, title=f"Training History: {metric_name}", line_shape=line_shape
            )
            fig.update_layout(
                title_y        = 1
                , xaxis_title  = "epochs"
                , yaxis_title  = metric_name
                , legend_title = None
                , height       = 400
                , template     = self.plot_template
                , yaxis        = dict(
                    side       = "right"
                    , tickmode = 'linear'
                    #when high loss, can't small fixed dtick. it freezes browser.
                    # `nticks` neither working here nor in `update_yaxes` below
                    , dtick    = tick_size
                )
                , legend = dict(
                    orientation = "h"
                    , yanchor   = "bottom"
                    , y         = 1.02
                    , xanchor   = "right"
                    , x         = 1
                )
            )
            fig.update_xaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
            fig.update_yaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
            if (call_display==True):
                fig.show()
            else:
                fig = set_dash_font(fig)
                figs.append(fig)
        if (call_display==False):
            return figs


    def confusion_matrix(
        self
        , cm_by_split
        , labels
        , call_display:bool = True
    ):
        figs = []
        for split, cm in cm_by_split.items():
            # change each element of z to type string for annotations
            cm_text = [[str(y) for y in x] for x in cm]

            fig = ff.create_annotated_heatmap(
                cm
                , x               = labels
                , y               = labels
                , annotation_text = cm_text
                , colorscale      = px.colors.sequential.BuGn
                , showscale       = False
                , colorbar        = dict(title='Count')
            )
            # add custom xaxis title
            fig.add_annotation(dict(
                font        = dict(color="white", size=15)
                , x         = 0.5
                , y         = 1.3
                , showarrow = False
                , text      = "Predicted Label"
                , xref      = "paper"
                , yref      = "paper"
            ))

            # add custom yaxis title
            fig.add_annotation(dict(
                font        = dict(color="white", size=15)
                , x         = -0.19
                , y         = 0.5
                # https://plotly.com/python/figure-structure/#positioning-with-paper-container-coordinates-or-axis-domain-coordinates
                , xref      = "x domain"
                , yref      = "paper"
                , showarrow = False
                , text      = "Actual Label"
                , textangle = -90

            ))

            fig.update_layout(
                title = dict(
                    text      = f"Confusion Matrix: {split.capitalize()}"
                    , y       = 0.0
                    , x       = 0.55
                    , xanchor = 'center'
                    , yanchor = 'bottom'
                )
                , template = self.plot_template
                , height   = 375  # if too small, it won't render in Jupyter.
                , yaxis    = dict(
                    tickmode     = 'linear'
                    , tick0      = 0.0
                    , dtick      = 1.0
                    , fixedrange = True#prevents zoom/pan
                    , tickfont   = dict(size=13)
                )
                , xaxis = dict(
                    categoryorder = 'category descending'
                    , fixedrange  = True#prevents zoom/pan
                    , tickfont    = dict(size=13)
                )
                #, margin = dict(r=75, l=150)#impacts y axis annotation x position
                , margin = dict(l=200)#impacts y axis annotation x position
            )

            fig.update_traces(
                hovertemplate = """predicted: %{x}<br>actual: %{y}<br>count: %{z}<extra></extra>"""
            )
            if (call_display==True):
                fig.show()
            else:
                fig = set_dash_font(fig)
                figs.append(fig)
        if (call_display==False):
            return figs


    def precision_recall(self, dataframe:object, call_display:bool=True):
        fig = px.line(
            dataframe
            , x     = 'recall'
            , y     = 'precision'
            , color = 'split'
            , title = 'Precision-Recall Curves'
        )
        fig.update_layout(
            legend_title = None
            , template   = self.plot_template
            , height     = 500
            , xaxis      = dict(title='Recall (Tₚ / Tₚ+Fₚ)')
            , yaxis      = dict(
                title      = 'Precision (Tₚ / Tₚ+Fₙ)'
                , side     = "right"
                , tickmode = 'linear'
                , tick0    = 0.0
                , dtick    = 0.05
            )
            , legend     = dict(
                orientation = "h"
                , yanchor   = "bottom"
                , y         = 1.02
                , xanchor   = "right"
                , x         = 1
            )
        )
        fig.update_xaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        fig.update_yaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        if (call_display==True):
            fig.show()
        else:
            fig = set_dash_font(fig)
            return fig


    def roc_curve(self, dataframe:object, call_display:bool=True):
        fig = px.line(
            dataframe
            , x     = 'fpr'
            , y     = 'tpr'
            , color = 'split'
            , title = 'Receiver Operating Characteristic (ROC) Curves'
        )
        fig.update_layout(
            template       = self.plot_template
            , legend_title = None
            , height       = 500
            , xaxis        = dict(
                title   = "False Positive Rate (FPR)"
                , tick0 = 0.00
                , range = [-0.025,1]
            )
            , yaxis      = dict(
                title      = "True Positive Rate (TPR)"
                , side     = "left"
                , tickmode = 'linear'
                , tick0    = 0.00
                , dtick    = 0.05
                , range    = [0,1.05]
            )
            , legend     = dict(
                orientation = "h"
                , yanchor   = "bottom"
                , y         = 1.02
                , xanchor   = "right"
                , x         = 1
            )
            , shapes=[
                dict(
                    type = 'line'
                    , y0=0, y1=1, x0=0, x1=1
                    , line = dict(dash='dot', width=2, color='#3b4043')
            )]
        )
        fig.update_xaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        fig.update_yaxes(zeroline=False, gridcolor='#2c3c4a', tickfont=dict(color='#818487'))
        if (call_display==True):
            fig.show()
        else:
            fig = set_dash_font(fig)
            return fig
    

    def feature_importance(
        self, feature_impacts:object, feature_id:int,
        permute_count:int, top_n:int,
        height:int, margin_left:int,
        call_display:bool
    ):
        if (top_n is not None):
            title = f"Feature Importance <sub>(feature.id:{feature_id}, permute_count:{permute_count}, top_n:{top_n})</sub><br><br>"
        elif (top_n is None):
            title = f"Feature Importance <sub>(feature.id:{feature_id}, permute_count:{permute_count})</sub><br><br>"
        
        fig = go.Figure()
        for feature, impacts in feature_impacts.items():
            fig.add_trace(go.Box(x=impacts, name=feature))
        fig.update_layout(
            template     = self.plot_template
            , height     = height
            , title      = title
            , showlegend = False
            , margin     = dict(l=margin_left)
        )
        fig.update_xaxes(
            title = f"Importance<br><sup>[permuted column loss - training loss]</sup>",
            # ticks not really showing.
            tickangle=45, nticks=15, gridcolor='#2c3c4a'
        )
        if (call_display==True):
            fig.show()
        else:
            fig = set_dash_font(fig)
            return fig

"""
These plots don't use the template because they are formatted for a light UI
"""
def confidence_binary(
    sigmoid_curve:object
    , point:object
    , height:int
    , labels:list
    , call_display:bool = True
):
    fig = px.line(
        sigmoid_curve
        , title      = f"<b>Probability of {labels[1]}</b>: {point['Probability'][0]*100:.1f}%"
        , x          = 'x'
        , range_x    = [-6, 6]
        , y          = 'y'
        , range_y    = [0, 1]
        , line_shape = 'spline'
    ).update_layout(
        title_x      = 0.04
        , title_y    = 0.90
        , margin     = dict(l=0, r=0, t=0, b=0)
        , height     = height
        , xaxis      = dict(title=None, showticklabels=False, fixedrange=True)
        , yaxis      = dict(title=None, showticklabels=False, fixedrange=True)
        , hoverlabel = dict(font=dict(size=15))
        , title      = dict(font=dict(family='Avenir',size=15))
    ).update_traces(
        mode            = 'lines'
        , line          = dict(width=2, color='#004473')
        , hovertemplate = None
        , hoverinfo     = 'skip'
        
    # Shade & divide the quadrants
    ).add_hrect(
        y0=0.5
        , y1=1.0
        , line_width=0
        , fillcolor="yellow"
        , opacity=0.1 
    ).add_shape(
        type         = "rect"
        , x0         = 0
        , x1         = -6
        , y0         = 0.5
        , y1         = 1.0
        , fillcolor  = 'white'
        , line_color = 'white'
        , opacity    = 0.7
    ).add_shape(
        type         = "rect"
        , x0         = 0
        , x1         = 6
        , y0         = 0.5
        , y1         = 0
        , fillcolor  = 'white'
        , line_color = 'white'
        , opacity    = 0.6
    ).add_hline(
        y           = 0.5
        , line_dash = "dash"
        , fillcolor = 'gray'
        , opacity   = 0.7

    # Plot the confidence point
    ).add_traces(
        list(
            px.line(
                point
                , x = 'Logit(Probability)'
                , y = 'Probability',
            ).update_traces(
                mode     = 'markers'
                , marker = dict(size=12, color='#004473')
            ).select_traces()
        )
    
    # Label the quadrants
    ).add_annotation(
        text        = f"{labels[0]}"
        , xref      = "paper"
        , yref      = "paper"
        , x         = 0.2
        , y         = 0.23
        , showarrow = False
        , font      = dict(size=18, color='#09243d')
    ).add_annotation(
        text        = f"{labels[1]}"
        , xref      = "paper"
        , yref      = "paper"
        , x         = 0.7
        , y         = 0.77
        , showarrow = False
        , font      = dict(size=18, color='#05300c')
    )
    if (call_display==True):
        fig.show()
    else:
        fig = set_dash_font(fig)
        return fig


def confidence_multi(
    label_probabilities:object
    , height:int
    , call_display:bool = True
):
    fig = px.pie(
        label_probabilities
        , values                  = 'Probability'
        , names                   = 'Labels'
        , hole                    = .7
        , title                   = f"<b>Probabilities:</b>"
        , color_discrete_sequence = px.colors.qualitative.Pastel
    ).update_traces(
        textposition = 'outside'
    ).update_layout(
        legend_title_text = "Labels:"
        , height          = height
        , title_x         = 0.1
        , title_y         = 0.7
        , margin          = dict(l=0, r=0, t=15, b=15)
        , title           = dict(font=dict(size=15))
        , legend          = dict(
            yanchor   = "top"
            , y       = 1
            , xanchor = "right"
            , x       = 1.3
            , title   = dict(font = dict(size=15))
        )
    )
    if (call_display==True):
        fig.show()
    else:
        fig = set_dash_font(fig)
        return fig
