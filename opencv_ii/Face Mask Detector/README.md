# File listing
- chart.png
  training progress - left to run over a weekend but peaked after just a few hours and 1500 iterations
- class.names         
  0 = face_with_mask, 1 = face_no_mask
- data_test.txt
  images used for validation and calculating mAP values
- data_train.txt
  images used for training
- yolov3-setup.data
  basic training configuration
- yolov3.cfg
  network architecture and training hyperparameters

# Main changes to yolov3.cfg
Subdivisions set to 64 to work with limited GPU RAM
Width/Height set to 832 to help detect small faces in test video (with limited success - this may actually have been counter productive)

# Darknet commands
## Command for training the network
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector train yolov3-setup.data yolov3.cfg darknet53.conv.74 -gpus 0,1 -map 2> train_log.txt

## Command for running inference on a test image
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector test yolov3-setup.data yolov3.cfg backup/yolov3_best.weights test_set\test-image1.jpg -thresh .6

## Command for trunning inference on a test video
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector demo yolov3-setup.data yolov3.cfg backup/yolov3_best.weights test_set\test-video2.mp4 -thresh .6 -out_filename out-vid2.avi -dont_show
