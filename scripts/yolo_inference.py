#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os
import cv2
import rospy  
from ultralytics import YOLO   
from dataclasses import dataclass

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import MultiArrayDimension

from hhcm_yolo_ros.msg import ObjectStatus
from hhcm_yolo_ros.srv import ClientToServerListString, ClientToServerListStringResponse

# This ROS node implements YOLO Inference 
# and publish the results on the corresponding topic  
@dataclass
class CameraIntrinsics:
    width: int = 0
    height: int = 0
    cx: float = 0.0
    cy: float = 0.0
    fx: float = 0.0
    fy: float = 0.0
    model: str = ''
    coeffs: list = None

    def __post_init__(self):
        if self.coeffs is None:
            self.coeffs = []

class YOLOInference():
    def __init__(self):
        # Initialize ROS node 
        self.node_name = 'yolo_inference'
        rospy.init_node(self.node_name, anonymous=False)

        self.rate = rospy.Rate(30)    
 
        # Get the parameters
        weights_path = rospy.get_param('~weights_path')
        weights_version = rospy.get_param('~weights_version')
        image_topic = rospy.get_param('~image_topic')
        if image_topic.endswith("compressed"):
            self.sub_compressed = True
            rospy.loginfo("Receiving compressed images")
        else:
            self.sub_compressed = False
            rospy.loginfo("Receiving raw images")

        camera_info_topic = rospy.get_param('~camera_info_topic')
        self.detection_confidence_threshold = rospy.get_param('~detection_confidence_threshold')
        self.cuda_device = rospy.get_param('~cuda_device')
        self.verbose = rospy.get_param('~verbose')
        self.visualize_boundbox = rospy.get_param('~visualize_boundbox')
        self.initial_classes = rospy.get_param('~initial_class')

        # initialize variables  
        self.what_to_perceive = []
        self._color_frame = []                       #store the color frame 
        self._intrinsics= CameraIntrinsics()         #store the camera info
        self.detection_response = []   
        self.bridge = CvBridge()
        self.image_received_header = Image.header

        # initialize subscribers 
        if self.sub_compressed:
            self.img_sub = rospy.Subscriber(image_topic, CompressedImage, self.getColorFrame)
        else:
            self.img_sub = rospy.Subscriber(image_topic, Image, self.getColorFrame)       

        self.image_received_header_seq = 0
        self.image_received_header_stamp = rospy.Time.now()
        self.image_received_header_frame_id = ""

        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.getCameraInfo)

        # initialize networks (loading weights) & useful variables 
        path_to_weights = os.path.join(weights_path, weights_version)
        self.model_detection = YOLO(path_to_weights) 
        # self.model_detection.to('cuda')

        self.model_detection_classes = self.model_detection.names # dictionary {id_number: "class_name"}

        if type(self.initial_classes) == list:
            self.what_to_perceive = self.initial_classes
        elif self.initial_classes == "*":
            self.what_to_perceive = list(self.model_detection_classes.values())
        elif self.initial_classes:
            if self.initial_classes in self.model_detection_classes.values():
                self.what_to_perceive = [self.initial_classes]
            else:
                rospy.logwarn(f"Initial class '{self.initial_classes}' not found in the model classes. Ignoring it.")

        # initialize publishers
        # 1. pub object_status
        self.object_status_pub = rospy.Publisher(self.node_name+'/object_status', ObjectStatus, queue_size =1) 
        """while self.object_status_pub.get_num_connections() == 0:
            rospy.loginfo("Inference: waiting for subscriber to connect.")
            self.rate.sleep()"""
            
        if self.visualize_boundbox:
            self.visualize_boundbox_pub = rospy.Publisher(
                self.node_name+'/visualize_boundbox/image_raw/compressed', CompressedImage, queue_size =1)
         
        # initialize services 
        # 1. start what_to_perceive_service
        self.what_to_perceive_service = rospy.Service(
            self.node_name+'/what_to_perceive', ClientToServerListString, self.what_to_perceive_service) 
        
        if self.verbose:
            rospy.loginfo(f"Available classes:\n{self.model_detection_classes}")
            rospy.loginfo(f"Selected classes:\n{self.what_to_perceive}")
         
    # Service callback 
    def what_to_perceive_service(self, srv):
        # This service is for receiving a list of "what" to perceive 
        self.what_to_perceive = []
        for wanted_cat in srv.data :
            if wanted_cat not in self.model_detection_classes.values():
                rospy.logwarn(f"Class '{wanted_cat}' not found in the model classes. Ignoring it.")
                continue
            self.what_to_perceive.append(wanted_cat)

        rospy.loginfo(f"Called what to perceive server: selected classes:\n{self.what_to_perceive}")
            
        if len(self.what_to_perceive) == 0:
            return ClientToServerListStringResponse(success=False)
        
        return ClientToServerListStringResponse(success=True)

    # Callback function to receive the color frame    
    def getColorFrame(self, msg):  
        try:
            if self.sub_compressed:
                self._color_frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                self._color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
            self.image_received_header_seq = msg.header.seq
            self.image_received_header_stamp = msg.header.stamp
            self.image_received_header_frame_id = msg.header.frame_id

        except CvBridgeError as e:
            rospy.logerr(e)
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
        self.camera_info_sub.unregister()

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
        results = self.model_detection(source=frame, show=False, save=False, verbose=False, device=self.cuda_device)[0] 
                
        if len(results)!= 0 : #case of predictions 

            confidence_list = results.boxes.conf.cpu().numpy()   
            detected_classes_list = results.boxes.cls.cpu().numpy().astype(int)

            if self.verbose: 
                rospy.loginfo(f"Detected classes ID: {detected_classes_list}")   
                detected_classes_name = [ self.model_detection_classes[i] for i in detected_classes_list ] 
                rospy.loginfo(f"Detected classes name: {detected_classes_name}")
                rospy.loginfo(f"Confidence list of detected classes: {confidence_list}")  
                
            try:
                xyxy = results.boxes.xyxy # left top corner (x1,y1) and right bottom corner (x2,y2)
                xyxy = xyxy.cpu().numpy()  
                
                xywh = results.boxes.xywh  # center (x,y), width and height of the bounding boxes 
                xywh = xywh.cpu().numpy()   

                if results.masks:
                    masks = results.masks.xy 
                else:
                    if self.verbose:
                        rospy.loginfo_once(10, "No masks results available. Are you sure you are running a Segmentation model?")
                    masks = []

            except: 
                rospy.logerr("Error while extracting bounding boxes and masks from inference results.")
                return []
            
             
            custom_pred =[]
            found_classes = {}

            for idx, conf in enumerate(confidence_list):
                id_class = detected_classes_list[idx]
                # print("id_class: ", id_class)
                # print("type(id_class): ", type(id_class))


                if conf >= self.detection_confidence_threshold:    
                    class_name = self.model_detection_classes[id_class]
                    
                    if class_name in object_classes_list:  
        
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

                        if len(masks) != 0:
                            shape_maks = masks[idx].shape
                            msg_object_status.segmentation_mask.data = masks[idx].flatten()
                            msg_object_status.segmentation_mask.layout.dim.append(MultiArrayDimension())
                            msg_object_status.segmentation_mask.layout.dim[0].label = 'rows'
                            msg_object_status.segmentation_mask.layout.dim[0].size = shape_maks[0]
                            msg_object_status.segmentation_mask.layout.dim[0].stride = shape_maks[1]
                            msg_object_status.segmentation_mask.layout.dim.append(MultiArrayDimension())
                            msg_object_status.segmentation_mask.layout.dim[1].label = 'cols'
                            msg_object_status.segmentation_mask.layout.dim[1].size = shape_maks[1]
                            msg_object_status.segmentation_mask.layout.dim[1].stride = 1

                        custom_pred.append(msg_object_status)  

                        self.object_status_pub.publish(msg_object_status)


            return custom_pred
        
        return []
    

    def object_detection(self, object_classes_list):
        # This function returns a list:
        # list[0]: boolean (True: some detections available, False: no detections)
        # list[1]: corresponding color frame of the prediction (necessary because in the meanwhile the frame may be updated)
        # list[2]: list of predictions (each item of this list is explained in the <detect> function)
         
        if (len(self._color_frame) > 0): 
            color_frame = self._color_frame
            # cv2.imshow("Image window", color_frame)
            # cv2.waitKey(3) 
            predictions = self.detect(frame=color_frame, object_classes_list=object_classes_list) 
            if len(predictions)!=0:
                results = [True, color_frame, predictions]  
                return results

        return [False, [], []]



    # Draw bounding box from YOLO Inference 
    def draw_boundbox_yolo(self):  

        if len(self.detection_response) != 0:

            image = self.detection_response[1]           
            if self.detection_response[0]:         

                predictions = self.detection_response[2]  # This is a list of msg_object_status

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
                    # TODO: change color depending on class
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, obj.object_class + " {:.2f}".format(obj.confidence), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                return self.bridge.cv2_to_compressed_imgmsg(image)
    
        return self.bridge.cv2_to_compressed_imgmsg(self._color_frame)


    def run_inference(self):
        # Loop function: at each cycle run the inference on the updated data

        # call object_detection 
        # self.detection_response[0] : bool of the result,  
        # self.detection_response[1] : color frame,  
        # self.detection_response[2] : list of predictions

        if len(self._color_frame) == 0:
            rospy.logwarn_throttle(2, "Received image is empty or no received at all!")
            return

        if len(self.what_to_perceive) != 0:
            self.detection_response = self.object_detection(self.what_to_perceive)

        if self.visualize_boundbox: 
            image = self.draw_boundbox_yolo()
            image.header.seq = self.image_received_header_seq
            image.header.stamp = self.image_received_header_stamp
            image.header.frame_id = self.image_received_header_frame_id
            
            self.visualize_boundbox_pub.publish(image)

        
    def run(self): 
        while not rospy.is_shutdown():  
            self.run_inference() 
            # rospy.loginfo("Running Inference")
            self.rate.sleep()

    
if __name__ == '__main__':

    node = YOLOInference()

    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
   