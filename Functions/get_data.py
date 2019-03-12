import serial
import argparse
import os
from time import sleep

def main(args):
	# initialize serial
	ser = serial.Serial(args.port, args.baudrate, timeout=1)

	ser.flushInput()
	ser.flushOutput()
	sleep(0.1)

	# write sequence 
	ser.write(bytes('N{0}P{1}G'.format(args.numiters, args.period), 'utf-8'))

	s = ""

	while(1):
		if ser.inWaiting():
			line = ser.readline()
			print(line)
			s += str(line).strip('b').strip("'")
			if 'END' in str(line):
				break

	data_start = "NOW.\n"
	data_end = "STOP"
	data_string = s[s.find(data_start) + len(data_start) : s.find(data_end)]

	data_string = data_string.strip('\\n').strip('\\r')

	#print(data_string)

	data_lines = data_string.split('\\r\\n')

	#print(data_lines)

	data_only_lines = [line.split('//')[1] for line in data_lines]

	fn = 'output/' + args.outputfile
	if(os.path.exists(fn + '_label.txt')):
		aw = 'a'
	else:
		aw = 'w'

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

	for line in data_only_lines:
		[accel, mag, gyro] = [l[4:] for l in line.split(';')]

		[a_x, a_y, a_z] = [a[2:] for a in accel.split(',')]
		[m_x, m_y, m_z] = [m[2:] for m in mag.split(',')]
		[g_r, g_p, g_y] = [g[2:] for g in gyro.split(',')]

		a_x_f.write(a_x + ' ')
		a_y_f.write(a_y + ' ')
		a_z_f.write(a_z + ' ')

		m_x_f.write(m_x + ' ')
		m_y_f.write(m_y + ' ')
		m_z_f.write(m_z + ' ')

		g_r_f.write(g_r + ' ')
		g_y_f.write(g_y + ' ')
		g_p_f.write(g_p + ' ')


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

	return 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='Hexiwear Data Acquisition')
	parser.add_argument("port", help="path to serial port", type=str)
	parser.add_argument("label", help="label for data", type=int)
	parser.add_argument("-b", "--baudrate", default=115200, help="baud rate", type=int)
	parser.add_argument("-o", "--outputfile", default="output", help="output file name prefix", type=str)
	parser.add_argument("-n", "--numiters", default=30, help="number of iterations", type=int)
	parser.add_argument("-p", "--period", default=0.1, help="period", type=float)

	args = parser.parse_args()

	main(args)