import dash
from dash import html, dcc, Input, Output

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("🧪 Upload Test - Minimal"),
    dcc.Upload(
        id='simple-upload',
        children=html.Div(['📁 Drag and Drop or Click to Select Files']),
        style={
            'width': '100%', 'height': '100px', 'lineHeight': '100px',
            'borderWidth': '2px', 'borderStyle': 'dashed',
            'borderRadius': '10px', 'textAlign': 'center',
            'margin': '20px', 'fontSize': '16px'
        },
        multiple=True
    ),
    html.Div(id='test-result', style={'margin': '20px', 'fontSize': '18px'})
])

@app.callback(
    Output('test-result', 'children'),
    Input('simple-upload', 'contents'),
    prevent_initial_call=True
)
def test_upload_callback(contents):
    if contents:
        return f"✅ SUCCESS! Received {len(contents)} file(s)"
    return "❌ No files detected"

if __name__ == '__main__':
    app.run_server(
        host='0.0.0.0',
        port=8051,
        debug=False,
        use_reloader=False
    )
