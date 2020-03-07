import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

LEN_TRACK = 32

def main():
	tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
	tracker_type = tracker_types[1]

	if tracker_type == 'KCF':
		tracker = cv2.TrackerKCF_create()
	if tracker_type == 'MEDIANFLOW':
		tracker = cv2.TrackerMedianFlow_create()	
	if tracker_type == 'MIL':
		tracker = cv2.TrackerMIL_create()

	video = cv2.VideoCapture(0)
	centroids = []
	if not video.isOpened():
		print("No video found")
		sys.exit()

	ret, frame = video.read()
	if ret:
		bbox = (154, 186, 230, 270)
		ret = tracker.init(frame, bbox)

		while True:
			ret, frame = video.read()
			if ret:
				timer = cv2.getTickCount()
				ret, bbox = tracker.update(frame)
				fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)

				if ret:
					color = (255,0,0)
					p1 = (int(bbox[0]), int(bbox[1]))
					p2 = (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]))
					cv2.rectangle(frame, p1, p2, color, 2, 1)
					centroids.append((p1[0]//2 + p2[0]//2,p1[1]//2+ p2[1]//2))
					
					if len(centroids) > LEN_TRACK:
						centroids = centroids[len(centroids)-LEN_TRACK:]
					
					for i in range(1,LEN_TRACK):
						if len(centroids)<=i or (centroids[i-1] is None or centroids[i] is None):
							continue
						thickness = int(np.sqrt(LEN_TRACK / float(i + 1)) * 2.5)
						cv2.line(frame, centroids[i - 1], centroids[i], color, thickness)

				else:
					cv2.putText(frame, "Tracking failur occured", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
				
				cv2.putText(frame, "Tracker: " + tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
				cv2.putText(frame, "FPS: " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

				cv2.imshow("Tracking", frame)

				k = cv2.waitKey(1) & 0xff
				if k==27:
					break


if __name__ == '__main__':
	main()