---
layout: default
---

NNonHexiwear is designed to be a repository that allows you to quickly set-up a work environment to build neural networks and run them on Hexiwear. Feel free to use the code as a basis for collecting data and implementing cool projects involving neural networks on Hexiwear. The process of setting up your system for development has been thoroughly documented in the following places:

| Click for README | Description |
| --------- | ----------- |
| [Functions/] | A script that acquires and labels time-series data from Hexiwear over serial (USB) |
| [HAR_TF_Example/] | An example of an implementation of a neural network for human activity recognition using Tensorflow |
| [HexiwearHARExample/] | An example/template project for implementing a neural network in Mbed for use on Hexiwear |

Slides can be found [here][slides].

## Introduction

### Computing on the edge

With the growth of the Internet of Things, cloud-connected embedded devices are everywhere. Machine learning and complex computation has traditionally been done on the cloud rather than on these less-powerful embedded systems (running on "the edge"), but many of these devices have become powerful enough to perform many operations locally before sending relevant data to the cloud. The benefits of computing on the edge are reduced latency, increased reliability, and sometimes better security.

### Mbed and Hexiwear

Mbed is a real-time operating system for embedded systems with ARM Cortex-M microcontrollers. It aids in development for these systems by providing a platform to quickly compile and run code. In our project, we chose Hexiwear, as it is Mbed-enabled and it is very sensor-rich. It has a total of 8 sensors: an accelerometer/magnetometer, gyroscope, pressure sensor (altitude), temperature/humidity sensor, ambient light sensor, and an optical heart rate sensor. It also includes Bluetooth Low Energy connectivity for wireless use. All of this fits in a compact, 2 inch by 2 inch package which can be worn on your wrist.

### Neural Networks (NNs)

Neural networks are computing systems inspired by the brains of animals, and are used in machine learning as classifiers. Neural networks are trained from labeled data, which often comes from sensors or images. Once trained, they can be used to classify new data with relatively high accuracy.

## Problem statement

Implementing a NN on the edge is a process that requires a lot of set-up and steps. For example, let's say you want to implement a NN on an embedded system with sensor data as input. First, you would have to obtain training data from the sensors. Next, you would have to set up an environment to run code to train and test your NN using the data. Lastly, you would then have to actually implement the NN on the system and have it take the correct input data.

Our goal in this project is to set up a work environment that has all the tools you need to gather data, train a NN, and implement it on an embedded system (Hexiwear). This environment will hopefully be used to streamline the implementation of cool projects using NNs on Hexiwear.

## Prior work

  1. [STMCube.AI]: Map and run pre-trained NNs on STM32 microcontrollers.
  2. [CMSIS-NN]: A collection of efficient NN kernels to maximize performance and minimize memory footprint in ARM Cortex-M processors.
  3. [uTensor]: A machine learning framework for Mbed and Tensorflow that converts a NN into C++ code.

## Technical Approach

We decided to use Hexiwear as our embedded system of choice. The overall process of implementing a NN on Hexiwear is as follows:

  - Obtain training data from sensors on Hexiwear via an Mbed project
  - Train a NN using the data in Tensorflow
  - Shrink the NN into C++ code using uTensor
  - Create an Mbed project for Hexiwear that includes the C++ NN code and inputs sensor data into the NN

These steps are described in more detail below.

### Data acquisition from Hexiwear

In this project, we focused on obtaining time-series IMU sensor data from Hexiwear. Data was pulled from the accelerometer (x, y, z), magnetometer (x, y, z), and gyroscope (roll, pitch, yaw). This was done by creating an Mbed program that captures the sensor data and sends it over serial. This program takes in input from serial that tells it the frequency at which to collect data, and how many points to collect. A Python script was also written to send requests for data to Hexiwear, and to parse the data that came back.

### Training a NN for Human Activity Recognition (HAR)

We decided to use UCI's HAR dataset, which consists of accelerometer and gyroscope data collected from Android phones. The dataset includes 3000 labeled examples, each of which is 128 time-series data points captured at 50 Hz. The gravity component is filtered out of the accelerometer data.

After importing the data, we used it to train a deep NN (DNN = all layers fully connected). Currently, the code is for a 4-layer DNN, but we also trained 2-layer DNNs as well. The resulting NN was output to a .pb file, which is what uTensor converts to C++ code. We also tried to use a LSTM/RNN, which yields better classification accuracy, but uTensor does not support the necessary functions for it.

### Shrinking and implementing

uTensor was used to shrink the .pb file into C++ code to be run on Hexiwear. This code was added to an Mbed project. In the main loop of the project, the accelerometer and gyroscope data is sampled at 50 Hz, and of these data points are passed as input to the uTensor NN, which outputs the expected label for the data. Unfortunately, we did not evaluate the accuracy of the uTensor implementation on Hexiwear.

## Future Directions

It would be nice to sample data from all the sensors, not just the IMU sensors, and make it even easier to collect training data. Adding BLE functionality would also make data collection nice due to wireless. In addition, when uTensor implements more functions, this project could be updated to include different types of NNs that are often better than just DNNs. Lastly, the project could also be changed to be more platform generic so that NNs could be quickly implemented for a variety of Mbed devices.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [slides]: <./NNonHexiwear_slides.pdf>
   [STMCube.AI]: <https://blog.st.com/stm32cubeai-neural-networks/>
   [CMSIS-NN]: <https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/new-neural-network-kernels-boost-efficiency-in-microcontrollers-by-5x>
   [uTensor]: <https://github.com/uTensor/uTensor>
   [Functions/]: <https://github.com/hisroar/NNonHexiwear/tree/master/Functions>
   [HAR_TF_Example/]: <https://github.com/hisroar/NNonHexiwear/tree/master/HAR_TF_Example>
   [HexiwearHARExample/]: <https://github.com/hisroar/NNonHexiwear/tree/master/HexiwearHARExample>
