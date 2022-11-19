# building_detection
 
In this repository, a novel vision-based method for detecting and confirming the target facade in the batch of neighboring buildings is introduced that performs matching using neural networks exploiting the texture features of facade segments; 
The network is trained using 9527 feature samples from 36 facade images.
As a result, a 79\% classification precision is achieved for trained network in correct facade detection and confirmation. 
Also, the success rate of overall entrance mission is found to be 13 out of 15 in real world experiments, provided the initial distance being less than 20 meters.

## Python Requirements

- tensorflow
- skimage
- cv2
- matplotlib

## Testing

Run the following command to run a test for a given building. To test other buildings, replace bld0 with any other folder in 'test_imgs' folder or create custom folder.

``` bash
python test.py bld0
```

