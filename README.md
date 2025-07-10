# A YOLO-ROS Package 
In this package you can find YOLO utilities to speed up your work with ROS !

## Dependencies 
- ultralytics
- json
- cv2
- numpy
- ros 

## How to run
- `roslaunch hhcm_yolo_ros yolo_inference.launch`, check arguments in the launch for customization
- Once running, a service is available to select the class to detect:
  ```
  rosservice call /yolo_inference/what_to_perceive "data:
  - '*'" 
  ```
  Use the '*' for activate the detection of all the classes of the loaded model.

## Resources
To train models you may find useful: https://github.com/ADVRHumanoids/hhcm_yolo_training