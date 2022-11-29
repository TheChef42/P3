#!/usr/bin/env python
"""
BlueRov video capture class
"""

import cv2
import gi
import numpy as np
from time import time

import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Import mavutil
from pymavlink import mavutil

# Create the connection
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
# Wait a heartbeat before sending commands
master.wait_heartbeat()


def look_at(tilt, roll=0, pan=0):
    """
    Moves gimbal to given position
    Args:
        tilt (float): tilt angle in centidegrees (0 is forward)
        roll (float, optional): pan angle in centidegrees (0 is forward)
        pan  (float, optional): pan angle in centidegrees (0 is forward)
    """
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
        1,
        tilt,
        roll,
        pan,
        0, 0, 0,
        mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING)
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            imageFrame = cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Smile", (sx, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
    return frame

class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
        latest_frame (np.ndarray): Latest retrieved video frame
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self.latest_frame = self._new_frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        array = np.ndarray(
            (
                caps_structure.get_value('height'),
                caps_structure.get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            np.ndarray: latest retrieved image frame
        """
        if self.frame_available:
            self.latest_frame = self._new_frame
            # reset to indicate latest frame has been 'consumed'
            self._new_frame = None
        return self.latest_frame

    def frame_available(self):
        """Check if a new frame is available

        Returns:
            bool: true if a new frame is available
        """
        return self._new_frame is not None

    def run(self):
        """ Get frame to update _new_frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        self._new_frame = self.gst_to_opencv(sample)

        return Gst.FlowReturn.OK


if __name__ == '__main__':
    # Create the video object
    # Add port= if is necessary to use a different one
    video = Video()

    # Capturing video through webcam
    webcam = video
    # webcam = WindowCapture('Pictures')

    # Start a while loop
    #loop_time = time()
    x = 1
    while (1):
        while True:
            # Wait for the next frame to become available
            if webcam.frame_available():
                # Only retrieve and display a frame if it's new
                imageFrame = webcam.frame()

                #cv2.imshow('frame', imageFrame)
                hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

                #gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)

                # calls the detect() function
                #canvas = detect(gray, imageFrame)

                # Set range for red color and
                # define mask
                red_lower = np.array([136, 87, 111], np.uint8)
                red_upper = np.array([180, 255, 255], np.uint8)
                red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

                # Set range for green color and
                # define mask
                green_lower = np.array([40, 52, 72], np.uint8)
                green_upper = np.array([80, 255, 255], np.uint8)
                green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

                # Set range for blue color and
                # define mask
                blue_lower = np.array([110, 50, 150], np.uint8)
                blue_upper = np.array([130, 255, 255], np.uint8)
                blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

                # Morphological Transform, Dilation
                # for each color and bitwise_and operator
                # between imageFrame and mask determines
                # to detect only that particular color
                kernal = np.ones((5, 5), "uint8")

                # For red color
                red_mask = cv2.dilate(red_mask, kernal)
                res_red = cv2.bitwise_and(imageFrame, imageFrame,
                                          mask=red_mask)

                # For green color
                green_mask = cv2.dilate(green_mask, kernal)
                res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                            mask=green_mask)

                # For blue color
                blue_mask = cv2.dilate(blue_mask, kernal)
                res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                           mask=blue_mask)

                # Creating contour to track red color
                contours, hierarchy = cv2.findContours(red_mask,
                                                       cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if (area > 300):
                        while True:
                            for angle in range(-50, 50):
                                look_at(angle * 100)
                                time.sleep(0.1)
                            for angle in range(-50, 50):
                                look_at(-angle * 100)
                                time.sleep(0.1)
                                break
                        x, y, w, h = cv2.boundingRect(contour)
                        imageFrame = cv2.rectangle(imageFrame, (x, y),
                                                   (x + w, y + h),
                                                   (0, 0, 255), 2)

                        cv2.putText(imageFrame, "Red Colour", (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 0, 255))
                        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)

                # Creating contour to track green color
                contours, hierarchy = cv2.findContours(green_mask,
                                                       cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if (area > 300):
                        x, y, w, h = cv2.boundingRect(contour)
                        imageFrame = cv2.rectangle(imageFrame, (x, y),
                                                   (x + w, y + h),
                                                   (0, 255, 0), 2)

                        cv2.putText(imageFrame, "Green Colour", (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 0))

                # Creating contour to track blue color
                contours, hierarchy = cv2.findContours(blue_mask,
                                                       cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                for pic, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if (area > 300):
                        x, y, w, h = cv2.boundingRect(contour)
                        imageFrame = cv2.rectangle(imageFrame, (x, y),
                                                   (x + w, y + h),
                                                   (255, 0, 0), 2)

                        cv2.putText(imageFrame, "Blue Colour", (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 0, 0))

                # Program Termination
                cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                x = 2
            # Allow frame to display, and check if user wants to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Reading the video from the
        # webcam in image frames
        #imageFrame = webcam.frame()
        # if x == 1:
        #    imageFrame = np.array(ImageGrab.grab())
        #    imageFrame = cv2.cvtColor(src=imageFrame, code=cv2.COLOR_BGR2RGB)
        # Convert the imageFrame in
        # BGR(RGB color space) to
        # HSV(hue-saturation-value)
        # color space
