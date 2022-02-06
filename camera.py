import cv2
from model import FacialExpressionModel
import numpy as np
import pandas as pd
import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import io

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #reading the Haar Cascade file
model = FacialExpressionModel("model.json", "model_weights.h5") #Passing the model json file and the weights to the FacialExpressionModel object
font = cv2.FONT_HERSHEY_SIMPLEX #Setting the font to the OpenCV

EMOTIONS_LIST = ["Angry", "Disgust","Fear", "Happy","Neutral", "Sad","Surprise"] #emotions encoding

def plot_preds(preds):
    '''
    A functions that takes in predictions and creates a file of the bar plot as a binary
    '''
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    sns.barplot(x = EMOTIONS_LIST,
                y = preds.reshape(7,),
                 ax=axis)
    output = io.BytesIO() #converting the chart to binary 
    FigureCanvas(fig).print_png(output)
    with open("charts/chart", "wb") as file:
        file.write(output.getvalue())

class VideoCamera(object):
    def __init__(self,logging,link=0):
        # link = r'./Face-Emotion-Recognition/videos/facial_exp.mkv'
        link = r'C:\Users\PC\Desktop\Final Code\final final code\videos\presidential_debate.mp4'
        self.video = cv2.VideoCapture(link)
        self.logging = logging
        self.log = pd.read_pickle(f'logs/{logging}.pkl')

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) #Converting the frame into gray scale
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48)) #Resizing the face into 48*48 to match our model
            preds = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis]) #get the prediction
            
            pred = EMOTIONS_LIST[np.argmax(preds)] #Getting the emotion name with the highest prediction/probability
            
            preds = preds.reshape(1,7)
            plot_preds(preds) #Plotting the prediction and saving the chart into the folder
            self.log = self.log.append(pd.DataFrame(list(preds), columns=EMOTIONS_LIST, index=[datetime.datetime.now()]), ignore_index=True)
            
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2) #Writing the prediction on the box
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2) #Drawing the rectangle box on the frame

        _, jpeg = cv2.imencode('.jpg', fr) #Encoding the frame as a jpg file
        pd.to_pickle(self.log, f'logs/{self.logging}.pkl') #Adding the predictions into a PKL file for future analyses
        return jpeg.tobytes() #Retruning the image as a bytes object