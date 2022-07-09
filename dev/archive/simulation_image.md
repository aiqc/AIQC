
# """
# Bah, forgot that you can't have 2 diff callbacks with the same output 
# """

# up = html.Div([
#     dcc.Upload(
#         ['Drag & Drop', html.I(' -or- '), html.A('Select Files', className='linx-upload')]
#         # Allow multiple files to be uploaded
#         , multiple  = False
#         , id        = 'upload-image'
#         , className = 'btn-upload'
#     ),
# ])       


# @callback(
#     [
#         Output('sim-preds-image', 'children'),
#         Output('sim-null', 'children'),
#     ],
#     Input('sim-btn-image', 'n_clicks'),
#     [
#         State('upload-image', 'contents'),
#         State('upload-image', 'filename'),
#         State('pred_dropdown', 'value'),
#         State('sim-preds', 'children'),
#     ]
# )
# def prediction_image(
#     n_clicks:int
#     , contents:list
#     , filenames:list
#     , model_id:int
#     , preds:list
# ):       
#     # Remember, n_clicks resets when changing model dropdown
#     if (n_clicks==0): 
#         raise PreventUpdate
#     else:
#         sim_null = None
#         if (preds is None):
#             preds = []
    
#     filename = filenames[0]
#     if ('.npy' in filename):
#         msg = "\nYikes - Only PIL supported images are supported in the UI.\n"
#         raise Exception(msg)
    
#     content_type, content_string = contents[0].split(',')
#     bytez = b64decode(content_string)
#     bytez = BytesIO(bytez)
#     img   = Image.open(bytez)
#     arr   = np.asarraay(img)

#     dims  = arr.ndim
#     # Don't `elif`
#     if (dims==2):
#         arr = np.array(arr)
#     if (dims==3):
#         arr = np.array(arr)

#     model   = Predictor.get_by_id(model_id)
#     feature = model.job.queue.splitset.features[0]
#     f_typs  = feature.get_dtypes()

#     # Retype the columns using the original dtypes
#     new_dset = Dataset.Image.from_array(arr4D_or_npyPath=arr, retype=f_typs)

#     # Generate the prediction using high-level api
#     prediction = mlops.Inference(
#         predictor      = model,
#         input_datasets = [new_dset]
#     )
    
#     # Information for the card body
#     queue = prediction.predictor.job.queue
#     label = queue.splitset.label
#     if (label is not None):
#         label = label.columns
#         if (len(label)==1): label=label[0]
#         label
#     else:
#         label = ""

#     analysis_typ = queue.algorithm.analysis_type
#     # Access the array, then the first prediction
#     sim_val = list(prediction.predictions.values())[0][0]
#     if ('regression' in analysis_typ):    
#         sim_txt = f"{label} = {sim_val:.3f}"
#     else:
#         sim_txt = f"{label} = {sim_val:}"

#     pred_id = prediction.id
#     # These preds are newly created. Impossible for them to be starred.
#     star = dbc.Button(
#         DashIconify(icon="clarity:star-line",width=20,height=20)
#         , id        = {'role':'pred_star','pred_id':pred_id}
#         , className = 'pred_star'
#         , color     = 'link'
#     )

#     sim_val   = html.Span(sim_txt, className='sim-val')
#     pred      = html.P([star, f"Prediction #{prediction.id}: ", sim_val], className="card-head")
#     pred      = dbc.Col(pred, width=9)

#     mod_id    = html.Span("Model ID: ", className='card-subhead')
#     mod_id    = html.P([mod_id, f"{model_id}"], className="card-text")
#     mod_id    = dbc.Col(mod_id, width=3) 

#     card_row  = dbc.Row([pred,mod_id], className='card-row')
#     card_body = [card_row]

#     if ('classification' in analysis_typ):
#         fig = prediction.plot_confidence(call_display=False)
#         fig = dcc.Graph(figure=fig, className='card-chart')
#         card_body.append(fig)

#     card = dbc.Card(
#         [
#             dbc.CardBody(card_body, className="card-bod"),
#             dbc.CardFooter(filename, className="card-fut"),
#         ],
#         className="sim-card"
#     )
#     preds.insert(0,card)
#     return preds, sim_null


btn-upload {
    position: relative;
    border-color: #62748e;
    border-radius: 10px;
    border-style: dashed;
    border-width: 2px;
    color: #899ab1;
    height: 60px;
    line-height: 60px;
    margin: 20px auto;
    text-align: center;
    width: 82%;
}

.linx-upload {
    color: #899ab1;
}

.linx-upload:hover {
    color: #ffda67;
}