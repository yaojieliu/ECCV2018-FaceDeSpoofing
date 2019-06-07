# Face De-Spoofing: Anti-Spoofing via Noise Modeling
Amin Jourabloo*, Yaojie Liu*, Xiaoming Liu

![alt text](http://www.cse.msu.edu/~liuyaoj1/images/caption_eccv18_git.png)

## Setup
Install the Tensorflow >=1.1, <2.0.

The source code files:
   1. "Architecture.py": Contains the architectures and the definitions of the loss functions.
   2. "data_train.py"  : Contains the functions for reading the training data.
   3. "Train.py"       : The main training file that read the training data, computes the loss functions and backpropagates error.
   4. "facepad-test.py": It performs the testing on the test videos and generates the score for each frame.

## Training
To run the training code:
source ~/tensorflow/bin/activate
python /data/train_demo/code/Train.py
deactivate

## Testing
To run the testing code on a test video ("Test_video.avi"):
1. python facepad-test.py -input Test_video.avi -isVideo 1
2. It will generate a txt file in the Score folder which contains the score for each frame.

## Acknowledge
Please cite the paper:

    @inproceedings{eccv18jourabloo,
        title={Face De-Spoofing: Anti-Spoofing via Noise Modeling},
        author={Amin Jourabloo*, Yaojie Liu*, Xiaoming Liu},
        booktitle={In Proceeding of European Conference on Computer Vision (ECCV 2018)},
        address={Munich, Germany},
        year={2018}
    }
    
    @inproceedings{eccv18jourabloo,
        title={Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision},
        author={Yaojie Liu*, Amin Jourabloo*, Xiaoming Liu},
        booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2018)},
        address={Salt Lake City, UT},
        year={2018}
    }

If you have any question, please contact: [Amin Jourabloo](amin.jourabloo@gmail.com) 
   
