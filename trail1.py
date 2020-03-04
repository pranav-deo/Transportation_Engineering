import tensorflow as tf
import tensornets as nets
import cv2
import numpy as np
import time

inputs = tf.placeholder(tf.float32, [None, 416,416,3])
model = nets.YOLOv3VOC(inputs)
CONFIDENCE_THRESH = 0

classes={'0':'person','1':'bicycle','2':'car','3':'bike','5':'bus','7':'truck'}
list_of_classes=[0,1,2,3,5,7]#to display other detected #objects,change the classes and list of classes to their respective #COCO indices available in their website. Here 0th index is for #people and 1 for bicycle and so on. If you want to detect all the #classes, add the indices to this list

with tf.Session() as sess:
	sess.run(model.pretrained())

	cap = cv2.VideoCapture("./JVLR_5mins.avi")
	no_of_frames = 0
	while(cap.isOpened()):
		_, frame = cap.read()
		image_shape = frame.shape
		frame = frame[:,(image_shape[1]-image_shape[0])/2:(image_shape[1]+image_shape[0])/2,:]

		img = cv2.resize(frame,(416,416))
		image = np.array(img).reshape(-1,416,416,3)
		start_time = time.time()
		preds = sess.run(model.preds, {inputs:model.preprocess(image)})
		
		# print("--- %s seconds ---" % (time.time() - start_time)) #to time it
		boxes = model.get_boxes(preds, image.shape[1:3])
		cv2.namedWindow('image',cv2.WINDOW_NORMAL)

		cv2.resizeWindow('image', 700,700)
		
		boxes1=np.array(boxes)
		no_of_frames =+ 1
		for j in list_of_classes: #iterate over classes
			count =0
			if str(j) in classes:
				lab=classes[str(j)]
			if len(boxes1) !=0:
			#iterate over detected vehicles
				for i in range(len(boxes1[j])): 
					box=boxes1[j][i] 
					#setting confidence threshold as 40%
					if boxes1[j][i][4]>=CONFIDENCE_THRESH: 
						count += 1    

						cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
						cv2.putText(img, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
			# print(lab,": ",count)
			if no_of_frames%500==0:
				print("Time Elapsed = %is\n"%no_of_frames//50)
				print(lab,": ",count)
		# #Display the output      
		cv2.imshow("image",img)  
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break