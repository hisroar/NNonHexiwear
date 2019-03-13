# NNonHexiwear

A workflow to help get started on implementing neural networks on Hexiwear using Mbed. [Hexiwear] is chosen because of its rich sensor ecosystem. The Hexiwear docking station is probably necessary to flash the binary files from Mbed.

### Included things

The following things are included in NNonHexiwear:

| Click for README | Description |
| --------- | ----------- |
| [Functions/] | A script that acquires and labels time-series data from Hexiwear over serial (USB) |
| [HAR_TF_Example/] | An example of an implementation of a neural network for human activity recognition using Tensorflow |
| [HexiwearHARExample/] | An example/template project for implementing a neural network in Mbed for use on Hexiwear |

### Getting started

NNonHexiwear allows you to implement neural networks on Hexiwear, but can probably be extended to work on other embedded systems.

A general workflow to implement a neural network on Hexiwear would be:
  1. Obtain data to train/test a neural network from Hexiwear sensors (or use existing data).
  2. Train a neural network in Tensorflow, and output a .pd file.
  3. Use [uTensor] command line interface to convert the .pd file to C++ code.
  4. Create an [Mbed] project including the necessary libraries for Hexiwear, and include the C++ code from uTensor.

If you need data, start with [Functions/] to begin collecting data.

If you already have data, check out [HAR_TF_Example/] to train a Neural Network.

Once you have a .pd file, try putting it on Hexiwear using [HexiwearHARExample/].

For guidance on how to set up the environment for running Tensorflow, check out [HAR_TF_Example/] For guidance on how to set up an Mbed project from scratch, check out [HexiwearHARExample/].

### Possible TODOs

 - Add more data to the data collection, and collect it without needing serial (BLE?).
 - Update for when uTensor implements more functions

License
----

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Hexiwear]: <https://www.hexiwear.com/>
   [uTensor]: <https://github.com/uTensor/uTensor>
   [Mbed]: <https://os.mbed.com/platforms/Hexiwear/>
   [Functions/]: <https://github.com/hisroar/NNonHexiwear/tree/master/Functions>
   [HAR_TF_Example/]: <https://github.com/hisroar/NNonHexiwear/tree/master/HAR_TF_Example>
   [HexiwearHARExample/]: <https://github.com/hisroar/NNonHexiwear/tree/master/HexiwearHARExample>
