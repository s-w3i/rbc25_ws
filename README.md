## Gripper control through ros2 service
```bash
ros2 run gripper_control gripper_control 
```
True to close the gripper
```bash
ros2 service call /gripper_control std_srvs/srv/SetBool "{data: true}"
```
False to open the gripper
```bash
ros2 service call /gripper_control std_srvs/srv/SetBool "{data: false}"
```

## ROS 2 whisper with OPEN AI API
This code only can be function while using the wheeltec mic
To bring up the mic and record voice:
```bash
ros2 launch wheeltec_mic_ros2 mic_init.launch.py 
```
REMARKS: program will only strat to record voice after receive awake command of 'hello hello'

To start speech recongition through OPENAI API
```bash
ros2 run voice_recognition voice_recognition_node
```
result of speech recognition will be published as '/speech_recognition_transcript'
