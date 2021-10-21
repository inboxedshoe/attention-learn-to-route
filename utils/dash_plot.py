
import dash
from dash import dcc
from dash import html
import networkx as nx
import plotly.graph_objs as go
import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json
from generate_data import generate_vrp_data

pts = generate_vrp_data(1,20)[0]


def generate_graph(pts):

    graph = nx.Graph()
    graph.add_nodes_from(pts[0])
    graph.add_nodes_from(pts[1])

    graph =nx.complete_graph(graph)

    return "graph"







external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'left',
        'color': colors['text']
    })

])


if __name__ == '__main__':
    app.run_server(debug=True)


