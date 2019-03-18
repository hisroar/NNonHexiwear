# An example Mbed project for implementing HAR on Hexiwear

Before trying this project out, it may be easier to get [this simple NN] working. It has a more thorough guide on how to setup an Mbed/uTensor project from scratch. Just replace `K66F` with `hexiwear` when compiling.

## HexiwearHARExample Quick-start

To get everything working, install the following:

  - [Python 3.7]
  - [Mbed-cli]: follow instructions on page to install.
  - [uTensor-cli]: `pip install utensor_cgen` should work.
  - [CoolTerm]: for serial communication

Once everything is installed, all you should need to do is:

```sh
make all
```

Then, just drag and drop the binary file located at `BUILD/HEXIWEAR/GCC_ARM-RELEASE/HexiwearHARExample.bin` to Hexiwear (`DAPLINK/`).

Configure CoolTerm (find the correct serial port and set baud rate to 115200), and press the reset button on the docking station. The collected sensor data should be printed, followed by the expected activity.

## Project overview

### main.cpp

#### CoolTerm

### Sensors

### Makefile


NOTE: in mbed_app.json, can allocate more stack space if you need more input data.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Mbed-cli]: <https://os.mbed.com/docs/mbed-os/v5.11/tools/installation-and-setup.html>
   [uTensor-cli]: <https://github.com/uTensor/utensor_cgen>
   [Python 3.7]: <https://www.python.org/downloads/>
   [CoolTerm]: <http://freeware.the-meiers.org/>
   [this simple NN]: <https://blog.hackster.io/simple-neural-network-on-mcus-a7cbd3dc108c>