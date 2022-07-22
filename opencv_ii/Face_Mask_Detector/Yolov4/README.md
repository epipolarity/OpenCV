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
1. <https://youtu.be/M7JhOhELhnE>
2. <https://youtu.be/xiHsL7_TS3Q>

## Main changes to yolov4.cfg
- Subdivisions set to 64 to work with limited GPU RAM  
- Width/Height set to 416 because using 832 when training yolov3 model didn't work as expected, so I hoped that using 416 would get the model used to working with small faces in low resolution images
- Burn_in set to 500 because I'd used 1000 when training yolo3v which seemed excessive and yet 100 seemed to be too small as the training seemed to become unstable and end up over-trained.
- Max_batches set to 5000 because having used 50,000 when training yolov3 found that the best solution was found much sooner. 
- Steps set to 4000 and 4500 with scales 0.1, 0.1. I didn't expect it to need to get this far into the training, but if it was still improving at this point then a reduction in learning rate might help it settle on an even slightly better result
- Classes set to 2 at each of three yolo layers
- Filters set to 21 at each of three conv layers immediately preceding a yolo layer: 21 = (classes + 5) * 3

## Darknet commands
### Command for training the network
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector train yolov4-setup.data yolov4.cfg yolov4.conv.137 -map 2> train_log.txt

### Command for running inference on a test image
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector test yolov4-setup.data yolov4.cfg backup/yolov4_best.weights test_set\test-image1.jpg -thresh .6

### Command for running inference on a test video
    ..\..\darknet-master\build\darknet\x64\darknet.exe detector demo yolov4-setup.data yolov4.cfg backup/yolov4_best.weights test_set\test-video1.mp4 -thresh .6 -out_filename out-vid1.avi -dont_show

