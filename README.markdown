# Introduction

Hi, here is part of my current project and I hope this would be helpful. Some of the code is from opencv samples or other open source projects. 
I will try my best to include the origin source for every tiny tool and please contact me if you find something wrong.

Development Environment: [opencv 2.3](https://launchpad.net/~gijzelaar/+archive/opencv2.3), Ubuntu 11.10 amd64, gcc 4.6.1
Please refer to [here](http://opencv.willowgarage.com/wiki/CompileOpenCVUsingLinux) for details

More information can be found on my [website](http://qiankanglai.me/2012/03/26/homography/)

# Implement detials

This code implements a tool for finding the homography between two similar pictures(input1.png & input2.png). SURF features are used to find pairs.
The result matrix can be used to "transform" input1 into input2! It means the result one should have the same "background". In other words, this can be used to combine several pictures together (to make a 360 degree photo, maybe~)

ps. This code is quiter simple and the main functions can be found in openCV. I make a wrap for it so this can be easily used in my whole project.
