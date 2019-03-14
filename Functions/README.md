# Hexiwear Functions

## Included

| Click for README | Description |
| --------- | ----------- |
| [ReadData/] | Read IMU sensor data from Hexiwear over serial (USB) |

### ReadData Quick-start

```sh
cd ReadData/
make compile
# flash Hexiwear with binary file located in:
# ./HexiwearReadData/BUILD/HEXIWEAR/GCC_ARM/HexiwearReadData.bin
# and press reset button to start program
python3 get_data.py /dev/tty.usbmodem[PORT] [label] # -h for help
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [ReadData/]: <https://github.com/hisroar/NNonHexiwear/tree/master/Functions/ReadData>