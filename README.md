# Application of GradCAM on DeepFake faces in videos

![gif1](gifs/gif1.gif)

It can also work with multiple faces.

![gif1](gifs/gif2.gif)

## Definition

[GradCAM](https://arxiv.org/abs/1610.02391) is a process that highlights in the form of a heatmap the areas of an input image that have triggered a prediction from a neural network.

## How does it work?

Using [dlib and OpenCV](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/), the faces in the video are detected. Then, the faces are sent to the [neural network](https://github.com/DariusAf/MesoNet) to find out if it is a DeepFake or not. Finally, if a DeepFake is found, we apply [GradCAM](https://github.com/cabjr/tf2cam) to highlight the areas that triggered the neural network prediction.

## What's in it?

The folder is composed of a folder containing the models used (like the *MesoInception_DF*), and three python files.


The file *classifiers.py* has not been modified in any way and comes from the [GitHub (DariusAf/MesoNet)](https://github.com/DariusAf/MesoNet) of [this paper](https://arxiv.org/abs/1809.00888). It contains the necessary architectures to load the models of the paper.


The file *gradcam.py* comes from the [GitHub cabjr/tf2cam](https://github.com/cabjr/tf2cam) and has been modified in parts to fit the problem. It is here that we explore the gradient of the model to generate the heatmap representing the area of of interest of the neural network.


The *main.py* file is the main file that must be launched to run everything. We coded it ourselves, except for some lines borrowed from [Adrian Rosebrock's pyimagesearch.com site](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) for the face detection.

The file also contains two videos from the [2020 Kaggle Deepfake Detection Challenge dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) and their two analyzed versions after being processed.

## Python libraries

The python libraries needed to run the code are *TensorFlow* (version 2), *OpenCV* (version 4), *dlib* (version 19), *numpy* and *imutils*.

## How to run it?

**To run the project**, you just have to change the name of the video at the very bottom of the *main.py* file and then run this file in an environment containing the python libraries mentioned above.


If you want to use the code in another context, you just have to change the way you retrieve the path of the video to be processed, to modify the loading of the model if necessary, create the object as at the end of the code in the *main.py* file, then call the method *.main()*.


```python
name_of_video = "video.mp4"
model_path = "models/MesoInception_DF.h5"

analyzer = VideoGanAnalyzer(name_of_video, model_path)
analyzer.main()
```

## Want to use another model?

If you want to use another model and that model does not have the same labels as the models I used, don't forget to change the GAN detection threshold part in the *gan_analysis()* function in *main.py*.
