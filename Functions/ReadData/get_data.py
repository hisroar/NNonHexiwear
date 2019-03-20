################################################################################
# get_data.py for data collection from Hexiwear
# 
# Written by Dennis Shim and Seraphine Goh. Email hisroar@gmail.com
#
# For usage instructions, run "python3 get_data.py -h".
#
# Designed to read accelerometer, magnetometer, and gyroscope data from Hexiwear
# over serial (USB). Assumes the data is formatted as follows:
# "[i]//acc x:[f],y:[f],z:[f];mag x:[f],y:[f],z:[f];gyr r:[f],p:[f],y:[f]"
# where [i] is the index (int) and [f] is a float representing the x/y/z or
# roll/pitch/yaw component of the sensor data.
#
# Saves data points in output files. Each output file represents one of the
# components (e.g. one for accelerometer x data, one for y, etc.). Also saves
# the label for the data (currently required to be an integer, change to str
# in argparse at the bottom if strings desired), and writes those to output
# files. Each new line in the output files (saved at directory "output/") should
# correspond to running one of the outputs
#
# Commented for clarity.
################################################################################

import serial
import argparse
import os
from time import sleep

def main(args):
    # initialize serial
    ser = serial.Serial(args.port, args.baudrate, timeout=1)

    # flush out serial buffer and wait for it to settle
    ser.flushInput()
    ser.flushOutput()
    sleep(0.1)

    # write sequence with number of data points to take and period at which
    # to take the data points, and also to start data acquisition.
    ser.write(bytes('N{0}P{1}G'.format(args.numiters, args.period), 'utf-8'))

    s = ""

    # loop until we see "END"
    while(1):
        if ser.inWaiting():
            line = ser.readline()
            # DEBUG: uncomment this line to see full serial output
            # print(line)
            s += str(line).strip('b').strip("'")
            if 'END' in str(line):
                break

    # only take string between start and end of data
    data_start = "NOW.\\n"
    data_end = "STOP"
    data_string = s[s.find(data_start) + len(data_start) : s.find(data_end)]

    # clean up data_string ends
    data_string = data_string.strip('\\n').strip('\\r')

    # split into lines, each line is one timestep's data
    data_lines = data_string.split('\\r\\n')

    # discard the index
    data_only_lines = [line.split('//')[1] for line in data_lines]

    OUTPUT = 'output/'
    if not os.path.exists(os.path.dirname(OUTPUT)):
    try:
        os.makedirs(os.path.dirname(OUTPUT))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

    # output file path + name. Probably assumes output/ path exists
    fn = OUTPUT + args.outputfile
    # if file doesn't exist, create new ("w"). Otherwise append ("a").
    if(os.path.exists(fn + '_label.txt')):
        aw = 'a'
    else:
        aw = 'w'

    # open files, yay.
    a_x_f = open(fn + '_accel_x.txt', aw)
    a_y_f = open(fn + '_accel_y.txt', aw)
    a_z_f = open(fn + '_accel_z.txt', aw)
    m_x_f = open(fn + '_mag_x.txt', aw)
    m_y_f = open(fn + '_mag_y.txt', aw)
    m_z_f = open(fn + '_mag_z.txt', aw)
    g_r_f = open(fn + '_gyro_r.txt', aw)
    g_y_f = open(fn + '_gyro_y.txt', aw)
    g_p_f = open(fn + '_gyro_p.txt', aw)
    label_f = open(fn + '_label.txt', aw)

    # loop through each line of data (line = timestep)
    for line in data_only_lines:
        # get strings for each sensor, cut off first 4 chars (e.g. ("acc "))
        [accel, mag, gyro] = [l[4:] for l in line.split(';')]

        # get strings for each data point, cut off first 2 chars (e.g. "x:")
        [a_x, a_y, a_z] = [a[2:] for a in accel.split(',')]
        [m_x, m_y, m_z] = [m[2:] for m in mag.split(',')]
        [g_r, g_p, g_y] = [g[2:] for g in gyro.split(',')]

        # now write float values to files
        a_x_f.write(a_x + ' ')
        a_y_f.write(a_y + ' ')
        a_z_f.write(a_z + ' ')

        m_x_f.write(m_x + ' ')
        m_y_f.write(m_y + ' ')
        m_z_f.write(m_z + ' ')

        g_r_f.write(g_r + ' ')
        g_y_f.write(g_y + ' ')
        g_p_f.write(g_p + ' ')

    # have to write newline b/c next line should be a new run
    a_x_f.write('\n')
    a_y_f.write('\n')
    a_z_f.write('\n')

    m_x_f.write('\n')
    m_y_f.write('\n')
    m_z_f.write('\n')

    g_r_f.write('\n')
    g_y_f.write('\n')
    g_p_f.write('\n')

    label_f.write(str(args.label) + '\n')

    # closing files is good practice
    a_x_f.close()
    a_y_f.close()
    a_z_f.close()

    m_x_f.close()
    m_y_f.close()
    m_z_f.close()

    g_r_f.close()
    g_y_f.close()
    g_p_f.close()

    label_f.close()

    return 0

# parse command line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("port", help="path to serial port", type=str)
    parser.add_argument("label", help="label for data", type=int)
    parser.add_argument("-b", "--baudrate", default=115200, help="baud rate", type=int)
    parser.add_argument("-o", "--outputfile", default="output", help="output file name prefix", type=str)
    parser.add_argument("-n", "--numiters", default=30, help="number of iterations", type=int)
    parser.add_argument("-p", "--period", default=0.1, help="period", type=float)

    args = parser.parse_args()

    main(args)