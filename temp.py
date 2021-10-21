
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
import torch

import matplotlib.pyplot as plt
import numpy as np

import pickle

# def generate_vrp_data_networkx(dataset_size, vrp_size):
#     CAPACITIES = {
#         10: 20.,
#         20: 30.,
#         50: 40.,
#         100: 50.
#     }
    
#     G = dict(enumerate(np.random.uniform(size=(vrp_size + 1 , 2))))  # Node locations))

   
#     np.random.randint(1, 10, size=(dataset_size, vrp_size))  # Demand, uniform integer 1 ... 9
#     # np.full(dataset_size, CAPACITIES[vrp_size]) # Capacity, same for whole dataset
    


def generate_graph(pts, instance = 0, dictionary = True):
    
    if not dictionary:        
        #temp demand list with depot and nodes    
        temp_demand = pts[instance][2]
        temp_demand.insert(0, 0)    
                    
        #temp demand list with depot and nodes    
        temp_pos = pts[instance][1]
        temp_pos.insert(0, pts[instance][0])    
        
    else:
        #temp demand list with depot and nodes    
        temp_demand = pts["demand"][instance].cpu().tolist()
        temp_demand.insert(0, 0) 
        
        #temp demand list with depot and nodes    
        temp_pos = pts["loc"][instance].cpu().tolist()
        temp_pos.insert(0, pts["depot"][instance].cpu().tolist())    
        
    graph = nx.complete_graph(len(temp_pos))
    
    for i in range(0, len(temp_pos)):
        graph.add_node(i, pos = temp_pos[i], demand = temp_demand[i])
                    
    
    #create fully_connected
    #graph = nx.complete_graph(graph)
    
    #set attributes
    #nx.set_node_attributes(graph, dict(enumerate(temp_demand) ), "demand")

    return graph

def create_plotly(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlOrRd',
            reversescale=False,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title='attention_prob',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    # #color
    # node_adjacencies = []
    # node_text = []
    # for node, adjacencies in enumerate(G.adjacency()):
    #     node_adjacencies.append(len(adjacencies[1]))
    #     node_text.append('# of connections: '+str(len(adjacencies[1])))
    
    node_trace.marker.color = np.exp(logs[0])
    node_trace.text = np.exp(logs[0])
    

    node_trace.marker.line.color = "black"
    node_trace.marker.symbol = ["square"]
    
    
    #draw
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                )
    
    fig.update_layout(
    width = 800,
    height = 800
    )
    
    fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    )
    
    return fig
   




with open('graph_res/cvrp50-original.pkl', 'rb') as handle:
    b = pickle.load(handle)

instance = 0

graphs = b[0]
logs = b[1][instance].cpu().numpy()
state = b[2][instance].cpu().numpy()

state = np.insert(state,0,0)

G = generate_graph(graphs, instance = instance, dictionary = True)
fig = create_plotly(G)


with open('graph_res/cvrp50-sparse.pkl', 'rb') as handle_1:
    b_1 = pickle.load(handle_1)

graphs_1 = b_1[0]
logs_1 = b_1[1][instance].cpu().numpy()
state_1 = b_1[2][instance].cpu().numpy()

state_1 = np.insert(state_1,0,0)

G_1 = generate_graph(graphs_1, instance = instance, dictionary = True)
fig_1 = create_plotly(G_1)


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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Transaction Network"

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

fig.update_layout(
    #plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='small Change',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'left',
        'color': colors['text']
    }),
    dcc.Graph(
    id='attention-graph',
    figure=fig
    ),
    
    html.Div(id='slider-output-container', style={
        'textAlign': 'left',
        'color': colors['text']
    }),
    
    dcc.Slider(
    id='my-slider',
    min=0,
    max = logs.shape[0]-1,
    step=1,
    value=1,
),
    
    dcc.Graph(
    id='sparse-graph',
    figure=fig_1
    ),
    
    html.Div(id='slider-output-container_1', style={
        'textAlign': 'left',
        'color': colors['text']
    }),
    
    dcc.Slider(
    id='my-slider_1',
    min=0,
    max = logs_1.shape[0]-1,
    step=1,
    value=1,
)


])



@app.callback(
    dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')], 
    [dash.dependencies.State('attention-graph', 'figure')])

def update_figure(value, figure):
    
    opacity = np.ones(logs.shape[1])
    opacity[state[:value]]=0.5
    
    marker_line_colors = np.repeat("black", logs.shape[1])
    marker_line_colors[state[value]] = "green"
    
    fig = go.Figure(figure)
    fig.update_traces(   
        #update opacity based on visited
        marker_opacity = opacity,        
                
        #update colors based on new log      
        marker_color = np.exp(logs[value]),
        
        #update marker line color based on current node
        marker_line_color = marker_line_colors,
        
        selector=dict(mode='markers')
    )

    fig.update_traces(
        text =  np.exp(logs[value])
    )
    
    return fig
    
@app.callback(
    dash.dependencies.Output('sparse-graph', 'figure'),
    [dash.dependencies.Input('my-slider_1', 'value')], 
    [dash.dependencies.State('sparse-graph', 'figure')])

def update_figure_1(value, figure):
    
    opacity = np.ones(logs_1.shape[1])
    opacity[state_1[:value]]=0.5
    
    marker_line_colors = np.repeat("black", logs_1.shape[1])
    marker_line_colors[state_1[value]] = "green"
    
    fig_1 = go.Figure(figure)
    fig_1.update_traces(   
        #update opacity based on visited
        marker_opacity = opacity,        
                
        #update colors based on new log      
        marker_color = np.exp(logs_1[value]),
        
        #update marker line color based on current node
        marker_line_color = marker_line_colors,
        
        selector=dict(mode='markers')
    )

    fig_1.update_traces(
        text =  np.exp(logs_1[value])
    )
    return fig_1
    
    # edge_x = []
    # edge_y = []
    # for edge in G.edges():
    #     x0, y0 = G.nodes[edge[0]]['pos']
    #     x1, y1 = G.nodes[edge[1]]['pos']
    #     edge_x.append(x0)
    #     edge_x.append(x1)
    #     edge_x.append(None)
    #     edge_y.append(y0)
    #     edge_y.append(y1)
    #     edge_y.append(None)
    
    # edge_trace = go.Scatter(
    #     x=edge_x, y=edge_y,
    #     line=dict(width=0.5, color='#888'),
    #     hoverinfo='none',
    #     mode='lines')
    
    
    # idx=[3, 7, 18]

    # xx=[]
    # yy=[]
    
    # for k in idx:
    #     xx.extend([fig['data'][0]['x'][3*k], fig['data'][0]['x'][3*k+1], None])
    #     yy.extend([fig['data'][0]['y'][3*k], fig['data'][0]['y'][3*k+1], None])
    
    # colored_edges=dict(type='scatter',
    #                    mode='line',
    #                    line=dict(width=2, color='red'),
    #                    x=xx,
    #                    y=yy)
    
    
    # data1=[colored_edges]+fig['data']
    # fig1=dict(data=data1, layout=fig['layout'])
    
    



@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    #dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')])

def update_output(value):
    return 'Step: {}'.format(value)

@app.callback(
    dash.dependencies.Output('slider-output-container_1', 'children'),
    #dash.dependencies.Output('attention-graph', 'figure'),
    [dash.dependencies.Input('my-slider_1', 'value')])

def update_output_1(value):
    return 'Step: {}'.format(value)




if __name__ == '__main__':
    app.run_server(debug=True)



# pos = pts[1]
# pos.insert(0, pts[0])

# nx.draw(G,pos)
# plt.show()



# # you want your own layout
# # pos = nx.spring_layout(G)
# pos = {point: point for point in pts[1]}

# # add axis
# fig, ax = plt.subplots()
# nx.draw(G, pos=pos, node_color='k', ax=ax)
# nx.draw(G, pos=pos, node_size=1500, ax=ax)  # draw nodes and edges
# nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
# # draw edge weights
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
# plt.axis("on")
# ax.set_xlim(0, 11)
# ax.set_ylim(0,11)
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# plt.show()
