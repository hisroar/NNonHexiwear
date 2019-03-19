# Hexiwear ReadData

## ReadData Quick-start

Tested on macOS. To get it working on Ubuntu, you will have to jump through some hoops to get Mercurial installed (need version >= 4.7, 4.5.2 is installed by default on Ubuntu 18.02).

To get everything working, install the following:

  - [Python 3.7]
  - [Mbed-cli]: follow instructions on page to install. Make sure you set up the `ARM_PATH`.

Once everything is installed, all you should need to do is:

```sh
cd ReadData/
mbed deploy
make compile
# flash Hexiwear with binary file located in:
# ./HexiwearReadData/BUILD/HEXIWEAR/GCC_ARM/HexiwearReadData.bin
# and press reset button to start program
pip install pyserial
python3 get_data.py /dev/tty.usbmodem[PORT] [label] # -h for help
```

## Description

### [HexiwearReadData/]

[HexiwearReadData/] contains an Mbed project for Hexiwear. It includes the necessary libraries to read data from the IMU sensors, as well as other libraries, which are all listed below.

| Library | Description |
| --------- | ----------- |
| FXAS21002 | Used to retrieve gyroscope sensor data |
| FXOS8700 | Used to retrieve accelerometer and magnetometer data |
| Hexi_KW40Z | (UNUSED) Used for Bluetooth low energy (BLE) connectivity |
| Hexi_OLED_SSD1351 | (UNUSED) Used to toggle LEDs on Hexiwear |
| MPL3155A2 | (UNUSED) Used to retrieve Barometric pressure and altitude |

The file [HexiwearReadData/main.cpp] contains the main loop code run on Hexiwear. It initializes the sensors and loops until it receives serial input from the user. The expected input string is detailed in comments in the code. It then samples time-series data from the sensors, and writes it back over serial.

The code has been commented for ease of reuse. To run on Hexiwear all one needs to do is `make compile` and then drag and drop the binary file `./HexiwearReadData/BUILD/HEXIWEAR/GCC_ARM/HexiwearReadData.bin` to Hexiwear.

### [get_data.py]

[get_data.py] is a Python script designed to retrieve time-series IMU sensor data from Hexiwear over serial (USB). It writes a string over serial to Hexiwear to begin data acquisition, and then reads in the formatted data, and parses it. It  writes the data to files in `output/`.

Also heavily commented for ease of reuse. Code is meant to be a starting point for more intensive data collection using Hexiwear. Feel free to repurpose or completely overhaul.

The command usage is described below:

```sh
python3 get_data.py [SERIAL_PATH] [LABEL] -b [BAUD_RATE] -o [OUTPUT_FILE] -n [NUM_ITERS] -p [PERIOD]
```

| Option | Description |
| --------- | ----------- |
| `SERIAL_PATH` | (REQUIRED) Path to the serial device. Usually something like `/dev/tty.usbmodem[NUMBER]`. |
| `LABEL` | (REQUIRED) Integer label for the data collected (assuming this will be used in a NN, label for data is usually helpful). |
| `BAUD_RATE` | (OPTIONAL) Serial baud rate, which can be changed in Hexiwear's main.cpp. Default `115200`. |
| `OUTPUT_FILE` | (OPTIONAL) Optional prefix for the output file name. Default `"output"`. |
| `NUM_ITERS` | (OPTIONAL) Number of data points to collect. Default `30`. |
| `PERIOD` | (OPTIONAL) Time period between data points in seconds. Default `0.1` (100 ms). |

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Mbed-cli]: <https://os.mbed.com/docs/mbed-os/v5.11/tools/installation-and-setup.html>
   [Python 3.7]: <https://www.python.org/downloads/>
   [HexiwearReadData/]: <https://github.com/hisroar/NNonHexiwear/tree/master/Functions/ReadData/HexiwearReadData>
   [get_data.py]: https://github.com/hisroar/NNonHexiwear/blob/master/Functions/ReadData/get_data.py
   [HexiwearReadData/main.cpp]: https://github.com/hisroar/NNonHexiwear/blob/master/Functions/ReadData/HexiwearReadData/main.cpp