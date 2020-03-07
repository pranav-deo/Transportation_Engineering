import time

from pydarknet import Detector, Image
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from random import randint
import numpy as np

LEN_TRACK = 10
to_track = False
first_iter = True
centroids = []

class Centroid_tracker():
    """docstring for Centroid_tracker"""
    def __init__(self):
        super(Centroid_tracker, self).__init__()
        self.centroids = []
        self.vehicle_number = None
        self.color = None

    def init_centroid(self, centroid, vehicle_number, color):
        self.centroids.append(centroid)
        self.vehicle_number = vehicle_number
        self.color = color

    def add_centroid(self, centroid):
        self.centroids.append(centroid)

def track_center(all_centroids, frame):
    for i in range(0, len(all_centroids)):
        if len(all_centroids[i].centroids) > LEN_TRACK:
            all_centroids[i].centroids = all_centroids[i].centroids[len(all_centroids[i].centroids)-LEN_TRACK:]

        # for i in range(1,LEN_TRACK):
        #     if len(centroids.centroids)<=i or (centroids.centroids[i-1] is None or centroids.centroids[i] is None):
        #         continue
        #     thickness = int(np.sqrt(LEN_TRACK / float(i + 1)) * 2.5)
        #     cv2.line(frame, centroids.centroids[i - 1], centroids.centroids[i], centroids.color, thickness)
        cv2.putText(frame, str(all_centroids[i].vehicle_number), (all_centroids[i].centroids[-1][0], all_centroids[i].centroids[-1][1]), cv2.FONT_HERSHEY_COMPLEX, 2, all_centroids[i].color)    

if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description='Process a video.')
    # parser.add_argument('path', metavar='video_path', type=str,
    #                     help='Path to source video')

    # args = parser.parse_args()
    # print("Source Path:", args.path)
    cap = cv2.VideoCapture("../mum.mp4")


    average_time = 0

    net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))

    multiTracker = cv2.MultiTracker_create()
    # colors = []
    all_centroids = {}

    no_of_frames_passed = 0
    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead
            frame = cv2.resize(frame,(1080,720))
            dark_frame = Image(frame)
            if not to_track:
                results = net.detect(dark_frame)
            del dark_frame

            # print("Total Time:", end_time-start_time, ":", average_time)

            counter = 0
            if not to_track:
                for cat, score, bounds in results:
                    color = (randint(0,255), randint(0,255), randint(0,255))
                    x, y, w, h = bounds
                    bbox = (int(x-w//2), int(y-h//2), int(w), int(h))
                    centroid = (int(x),int(y))
                    centroid_tracker = Centroid_tracker()
                    centroid_tracker.init_centroid(centroid, counter, color)
                    all_centroids[counter] = (centroid_tracker)
                    multiTracker.add(cv2.TrackerKCF_create(), frame, bbox)
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),color)
                    cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, color)
                    to_track = True
                    counter += 1

            elif to_track:
                    success, boxes = multiTracker.update(frame)
                    for i, newbox in enumerate(boxes):
                        x,y,w,h = newbox
                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w),int(y+h)),all_centroids[i].color)
                        centroid = (int(x+w//2),int(y+h//2))
                        all_centroids[i].add_centroid(centroid)
                    track_center(all_centroids, frame)


            counter = 0
            end_time = time.time()
            average_time = average_time * 0.8 + (end_time-start_time) * 0.2
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)
            
            print("FPS: ", fps)

            cv2.imshow("preview", frame)
            no_of_frames_passed += 1
        else:
            print(frame)


        # k = cv2.waitKey(1)
        # if k == 0xFF & ord("q"):
        #     break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
