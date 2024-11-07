#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
 
import cv2
import rospy  
import numpy as np  
from ultralytics import YOLO  
import json 

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, Range 

from hhcm_yolo_ros.msg import ObjectStatus
from hhcm_yolo_ros.srv import ClientToServerString, ClientToServerStringResponse

# This ROS node implements YOLO Inference 
# and publish the results on the corresponding topic 
 

class YOLOInference():
    def __init__(self):
        # Initialize ROS node 
        rospy.init_node('yolo_inference', anonymous=False)

        self.rate = rospy.Rate(30)    
 

        #initialize variables  
        self.what_to_perceive = []
        self._color_frame = []                       #store the color frame
        self._tof = []                               #store the TOF
        self._intrinsics= []                         #store the camera info
        self.detection_response = []   

        #initialize subscribers 
        self.img_sub = rospy.Subscriber('/nicla/camera/image_raw', Image, self.getColorFrame)
        self.camera_info_sub = rospy.Subscriber('/nicla/camera/camera_info', CameraInfo, self.getCameraInfo)
        self.tof_sub = rospy.Subscriber('/nicla/tof', Range, self.getTOF)


        #initialize networks (loading weights) & useful variables
        PATH_WEIGHTS = '../weights/'

        # model for object instance segmentation
        self.model_detection = YOLO(PATH_WEIGHTS+"yolo11n-seg.pt")  
        # self.model_detection.to('cuda')

        # model classes 
        self.model_detection_classes = self.model_detection.names # dictionary {id_number: "class_name"}
         

        # detection threshold 
        self.detection_confidence_threshold = 0.5

        # initialize publishers
        # 1. pub object_status
        self.object_status_pub = rospy.Publisher('/yolo/object_status', ObjectStatus, queue_size =10) 
        """while self.object_status_pub.get_num_connections() == 0:
            rospy.loginfo("Inference: waiting for subscriber to connect.")
            self.rate.sleep()"""
         

        # initialize services 
        # 1. start what_to_perceive_service
        self.what_to_perceive_service = rospy.Service('/yolo/what_to_perceive', ClientToServerString, self.what_to_perceive_service) 
         

    # Service callback 
    def what_to_perceive_service(self, msg):
        # This service is for receiving a list of "what" to perceive 
        self.what_to_perceive = msg.data
        self.what_to_perceive = json.loads(self.what_to_perceive)

        return ClientToServerStringResponse(success=True)

    # Callback function to receive the color frame    
    def getColorFrame(self, msg):  
        bridge = CvBridge()
        try:
            self._color_frame = bridge.imgmsg_to_cv2(msg, "bgr8") 
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("Image window", self._color_frame)
        # cv2.waitKey(3)

    
    # Callback function to receive the camera info
    def getCameraInfo(self, cameraInfo):  
        self._intrinsics.width = cameraInfo.width
        self._intrinsics.height = cameraInfo.height
        self._intrinsics.cx = cameraInfo.K[2]
        self._intrinsics.cy = cameraInfo.K[5]
        self._intrinsics.fx = cameraInfo.K[0]
        self._intrinsics.fy = cameraInfo.K[4]
        self._intrinsics.model  = cameraInfo.distortion_model
        self._intrinsics.coeffs = [i for i in cameraInfo.D]  

    
    # Callback function to receive TOF
    def getTOF(self, tof_msg):  
        self._tof.range = tof_msg.range 
        self._tof.min_range = tof_msg.min_range 
        self._tof.max_range = tof_msg.max_range 

    
    # Inference functions 
    def detect(self, frame, object_classes_list): 
        # This function calls the model_detection for instance segmentation
        # and publishes the detections on the corresponding ROS topics.
        # It returns a customized list of items/predictions (e.g [item1, item2, item3,...]) arranged as:
        # item[0]: center [x,y] of a box (i.e the bounding box surrounding the detected object)
        # item[1]: width and height [w,h] of a box 
        # item[2]: class in string (e.g. "lever")
        # item[3]: confidence of the prediction
        # item = [[x,y], [w,h], "class", conf]
        # In the returned list there will be only predictions with a confidence above a threshold (self.detection_confidence_threshold).
        
        # YOLO function to predict on a frame using the loaded model
        results = self.model_detection(source=frame, show=False, save=False, verbose=False, device=0)[0] 
        
        if len(results)!= 0 : #case of predictions 

            confidence_list = results.boxes.conf.cpu().numpy()   
            # print("confidence_list: ", confidence_list)
            detected_classes_list = results.boxes.cls.cpu().numpy().astype(int)
            #print("detected_classes_list: ", detected_classes_list)      
             
            custom_pred =[]
            found_classes = {}

            for idx, conf in enumerate(confidence_list):
                id_class = detected_classes_list[idx]
                #print("id_class: ", id_class)
                #print("type(id_class): ", type(id_class))


                if conf >= self.detection_confidence_threshold:    
                    class_name = self.model_detection_classes[id_class]
                    
                    if class_name in object_classes_list:  

                        try:    
                            xyxy = results.boxes.xyxy # left top corner (x1,y1) and right bottom corner (x2,y2)
                            xyxy = xyxy.cpu().numpy() 
                            xywh = results.boxes.xywh  # center (x,y), width and height of the bounding boxes 
                            xywh = xywh.cpu().numpy()  
                        except: 
                            print("Error while extracting bounding box from inference results.")
                            return []
        
                        # initialize message corresponding to objects instance segmentation & detection 
                        msg_object_status = ObjectStatus()   

                        if id_class in found_classes: 
                            available_obj_id = found_classes[id_class]
                            found_classes[id_class] += 1
                        else: 
                            found_classes[id_class] = 1
                            available_obj_id = 0

                        msg_object_status.object_class = class_name 
                        msg_object_status.object_ID = available_obj_id 
                        msg_object_status.confidence = conf 
                        msg_object_status.bounding_box_vertices = [xyxy[idx][0],xyxy[idx][1], xyxy[idx][2], xyxy[idx][3]] 
                        msg_object_status.bounding_box_center = [xywh[idx][0], xywh[idx][1]] 
                        msg_object_status.bounding_box_vertices_meter = [(xyxy[idx][0]-self._intrinsics.cx)/self._intrinsics.fx,(xyxy[idx][1]-self._intrinsics.cy)/self._intrinsics.fy, (xyxy[idx][2]-self._intrinsics.cx)/self._intrinsics.fx, (xyxy[idx][3]-self._intrinsics.cy)/self._intrinsics.fy] 
                        msg_object_status.bounding_box_center_meter = [(xywh[idx][0]-self._intrinsics.cx)/self._intrinsics.fx, (xywh[idx][1]-self._intrinsics.cy)/self._intrinsics.fy] 
                        msg_object_status.bounding_box_wh = [xywh[idx][2], xywh[idx][3]] 
                        msg_object_status.segmentation_mask = None # TODO 
                        msg_object_status.pose = None # TODO: maybe another module for this ? 

                        item = [msg_object_status]
                        custom_pred.append(item)  

                        self.object_status_pub.publish(msg_object_status)
            return custom_pred
        
        return []
    

    def object_detection(self, object_classes_list):
        # This function returns a list because of the pose estimation module:
        # list[0]: boolean (True: some detections available, False: no detections)
        # list[1]: corresponding color frame of the prediction (necessary because in the meanwhile the frame may be updated)
        # list[2]: corresponding tof (Note: the tof is of the image center, not of the predictions)
        # list[3]: list of predictions (each item of this list is explained in the <detect> function)
 
        if (len(self._color_frame) > 0) and (len(self._tof) > 0):
            color_frame = self._color_frame
            # cv2.imshow("Image window", color_frame)
            # cv2.waitKey(3)
            tof = self._tof 
            predictions = self.detect(frame=color_frame, object_classes_list=object_classes_list) 
            if len(predictions)!=0:
                results = [True, color_frame, tof, predictions]  
                return results

        return [False, [], [], []]



    # Draw bounding box from YOLO Inference 
    def draw_boundbox_yolo(self, image):  
        response = self.detection_response 
        if len(response) != 0 and response[0]:         

            predictions = response[3]  # This is a list of msg_object_status
            
            for obj in predictions:  
                center_x = obj.bounding_box_center[0]
                center_y = obj.bounding_box_center[1]
                width = obj.bounding_box_wh[0]
                height = obj.bounding_box_wh[1]

                # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)

                # Draw the rectangle on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image



    def run_inference(self):
        # Loop function: at each cycle run the inference on the updated data

        # call object_detection 
        # response[0] : bool of the result,  
        # response[1] : color frame, 
        # response[2] : TOF, 
        # response[3] : list of predictions

        if "yolo" in self.what_to_perceive:
            self.detection_response = self.object_detection(self.what_to_perceive['yolo'])
 

        
    def run(self): 
        while not rospy.is_shutdown():  
            self.run_inference() 
            # rospy.loginfo("Running Inference")
            self.rate.sleep()

        
def main():
    try:
        node = YOLOInference()
        node.run()
    except rospy.ROSInterruptException:
        pass
     

if __name__ == '__main__':
    main()
   
 