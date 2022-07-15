import cv2
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import deque

from flask import Flask, Response

from dash import Dash, html, dcc
from flask import Flask

import dash_bootstrap_components as dbc

from mainTracker import Tracker,vis_track,draw_lines,lines

server = Flask(__name__)

app = Dash(__name__,server = server, external_stylesheets=[dbc.themes.BOOTSTRAP])

tracker = Tracker(filter_classes=None, model='yolox-l',ckpt='weights/yolox_l.pth')

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
        

def gen(camera):
    fps = 0.0
    while True:
        frame = camera.get_frame()
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



header = dbc.Col(width=10,
    children=[ html.H1("Traffic flow Management", style ={'text-align':'center'})]
)
header = dbc.Col(width=10,
    children=[ html.H1("Traffic flow Management", style ={'text-align':'center'})]
)


app.layout = html.Div([
    dbc.Row([]),#Header
    dbc.Row([]),#Row
    dbc.Row([videofeeds]),#VideoFeed and 2 graphs

])






if __name__ =='__main__':
    app.run_server(port = 8050)