import cv2
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import deque

from flask import Flask, Response
import plotly.graph_objects as go
from dash import Dash, html, dcc,Input,Output
from flask import Flask

import dash_bootstrap_components as dbc

from mainTracker import Tracker,vis_track,draw_lines,lines

server = Flask(__name__)

app = Dash(__name__,server = server, external_stylesheets=[dbc.themes.BOOTSTRAP])

tracker = Tracker(model='yolox-l',ckpt='weights/yolox_l.pth')

Main = deque(maxlen =1000)

# _____________________ Video feeds___________________

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class VideoCamera(object):
    def __init__(self):
        global res;
        self.video = cv2.VideoCapture(sys.argv[1])
        res = f"{int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))}" 

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        
        success, image = self.video.read()
        
        

           
        if success:
            image =draw_lines(lines,image)
            image,bbox,data = tracker.update(image,logger_=False)
            image=vis_track(image,bbox)
            Main.extend(data)

            ret,buffer=cv2.imencode('.jpg',image)
            image=buffer.tobytes()
        return image
        
fps = 0.0
res = "A x B"
stream = "Stream 1"
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        fps =fps+1
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

videofeed = html.Img(src="/video_feed")

videofeeds = dbc.Col(width=4, children =[
    html.Img(src="/video_feed", style={
        'max-width':'100%',
        'height':'auto',
        'display':'block',
        'margin-right':'auto'
        })
    ]   
)

def create_card(Header,Value,cardcolor):
    card=dbc.Col([
        dbc.Card([
            dbc.CardHeader(Header,style={'text-align':'center'}),
            dbc.Cardbody([
                html.H3(Value,style={'text-align':'center'})
            ])
        ],color=cardcolor,inverse=True,style = {
            "width":"18rem",
            'text-align':'center',
            "vertical-align":"middle"
        })
        ])
    return card
    


header = dbc.Col(width=10,
    children=[ html.H1("Traffic flow Management", style ={'text-align':'center'})]
)
figure1=dbc.Col([dcc.Graph(id='live-graph1')],width=4)
figure2=dbc.Col([dcc.Graph(id='live-graph2')],width=4)
@app.callback(
    [
        Output('live-graph1','figure'),
        Output('live-graph2','figure'),
        Output('cards','children'),
    ],
    [
        Input('visual-update','n_intervals')
    ]
)
def update_visuals():
    fig1 = go.FigureWidget()
    fig2 = go.FigureWidget()


    vehicleslastminute=0
    vehiclestotal=0
    if len(df)!=0:
        df = pd.DataFrame(Main)
        df = df.pivot_table(index=['Time'],
        columns = 'Category',aggfunc = {'Category:"count'}).fillna(0)
        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        df.Time = pd.to_datetime(df.Time)
        columns = list(df.columns)
        columns.remove('Time')
        values_sum = []
        for col in columns:
            fig1.add_scatter(name=col,x=df['Time'],y=df[col],fill = 'tonexty', line_shape='spline')
            fig2.add_scatter(name=col,x=df['Time'],y=df[col].cumsum(),fill = 'tonexty', line_shape='spline')
            vehicleslastminute+=df[col].values[-1]
            vehiclestotal+=df[col].cumsum().values[-1]
            values_sum.append(df[col].sum())
    cards = [
        create_card(Header="Vehicles This Minute",Value = vehicleslastminute,cardcolor="primary"),
        create_card(Header="Total Vehicles",Value = vehiclestotal,cardcolor="info"),

        create_card(Header="Frame",Value = fps,cardcolor="secondary"),
        create_card(Header="Resolution",Value = res,cardcolor="warning"),
        create_card(Header="Stream",Value = stream,cardcolor="danger"),

    ]
    return fig1,fig2






app.layout = html.Div([
    dcc.Interval(id='visual-update',n_intervals=0),
    dbc.Row([header]),#Header
    dbc.Row(id="cards"),#Row
    dbc.Row([videofeeds,figure1,figure2])#VideoFeed and 2 graphs

])






if __name__ =='__main__':
    app.run_server(port = 8050)