
import torch
import numpy as np
import cv2
from time import time

from functorch._src.aot_autograd import model_name


class LegoDetection:


    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cpu'

    def get_video_capture(self):

        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):

        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
            print('1')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            print('2')
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, : -1]
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n=len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row=cord[i]
            if row[4]>= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def __call__(self):

        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (640, 640))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.imshow("Multiple Color Detection in Real-TIme", frame)
            if cv2.waitKey(4) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        cap.release()

detector = LegoDetection(capture_index=0, model_name='./runs/train/yolov5s_results3/weights/best.pt')
detector()








""""""
#Example of how to connect pymavlink to an autopilot via an UDP connection


# Disable "Bare exception" warning
# pylint: disable=W0702

# Import mavutil

from roboflow import Roboflow
rf = Roboflow(api_key="n5mOz2UBdtv07UIFuhIF")
project = rf.workspace("newoysters").project("oysters-4vutw")
dataset = project.version(41).download("yolov5")

import glob
from IPython.display import Image, display
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

from IPython.core.magic import register_line_cell_magic
#@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))

from utils.plots import plot_results

Image(filename='./runs/train/exp2/results.png', width=1000)  # view results.png





for imageName in glob.glob('./runs/detect/exp2/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
# Create the connection
#  If using a companion computer
#  the default connection is available
#  at ip 192.168.2.1 and the port 14550
# Note: The connection is done with 'udpin' and not 'udpout'.
#  You can check in http:192.168.2.2:2770/mavproxy that the communication made for 14550
#  uses a 'udpbcast' (client) and not 'udpin' (server).
#  If you want to use QGroundControl in parallel with your python script,
#  it's possible to add a new output port in http:192.168.2.2:2770/mavproxy as a new line.
#  E.g: --out udpbcast:192.168.2.255:yourport
""""""