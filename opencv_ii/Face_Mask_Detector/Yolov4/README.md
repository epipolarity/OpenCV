# YOLOv4 submission

## File listing
- **predictions**  
  results of inference on the test images
- **README.md**  
  this file
- **chart.png**  
  training progress - 
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

  
## Results of inference on test videos
1. <https://youtu.be/xxxxxxxxx>
2. <https://youtu.be/xxxxxxxxx>

## Main changes to yolov3.cfg
- Subdivisions set to 64 to work with limited GPU RAM  
- Width/Height set to 
- Burn_in set to 500 because 
- Max_batches set to 5000 because 
- Steps set to
- Classes set to 2 at each of three yolo layers
- Filters set to 21 at each of three conv layers immediately preceding a yolo layer: 21 = (classes + 5) * 3

## Darknet commands
### Command for training the network
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector train yolov4-setup.data yolov4.cfg yolov4.conv.137 -map 2> train_log.txt

### Command for running inference on a test image
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector test yolov4-setup.data yolov4.cfg backup/yolov4_best.weights test_set\test-image1.jpg -thresh .6

### Command for running inference on a test video
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector demo yolov4-setup.data yolov4.cfg backup/yolov4_best.weights test_set\test-video1.mp4 -thresh .6 -out_filename out-vid1.avi -dont_show

