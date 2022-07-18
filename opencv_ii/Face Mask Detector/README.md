# File listing
- **chart.png**  
  training progress - left to run over a weekend but peaked after just a few hours and 1500 iterations
- **class.names**      
  0 = face_with_mask, 1 = face_no_mask
- **data_test.txt**  
  images used for validation and calculating mAP values
- **data_train.txt**  
  images used for training
- **yolov3-setup.data**  
  basic training configuration
- **yolov3.cfg**  
  network architecture and training hyperparameters

# Main changes to yolov3.cfg
- Subdivisions set to 64 to work with limited GPU RAM  
- Width/Height set to 832 to help detect small faces in test video (with limited success; this may actually have been counter productive)
- Burn_in set to 1000 because time was not an issue and had not had success previously using a value of 100
- Max_batches set to ~50000 because I planned to leave this running as long as it took and had not had success previously after 10000 batches
- steps set to high values which were never reached on this training run which i terminated after 48 hours at 25000 batches
- classes set to 2 at each of three yolo layers
- filters set to 21 at each of three conv layers immediately preceding a yolo layer: 21 = (5 + classes) * 3

# Darknet commands
## Command for training the network
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector train yolov3-setup.data yolov3.cfg darknet53.conv.74 -gpus 0,1 -map 2> train_log.txt

## Command for running inference on a test image
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector test yolov3-setup.data yolov3.cfg backup/yolov3_best.weights test_set\test-image1.jpg -thresh .6

## Command for trunning inference on a test video
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector demo yolov3-setup.data yolov3.cfg backup/yolov3_best.weights test_set\test-video2.mp4 -thresh .6 -out_filename out-vid2.avi -dont_show
