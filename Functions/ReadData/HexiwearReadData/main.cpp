////////////////////////////////////////////////////////////////////////////////
// main.cpp for data collection from Hexiwear
//
// Written by Dennis Shim and Seraphine Goh. Email hisroar@gmail.com
//
// Designed for use over serial (USB) connection to Hexiwear. Samples data from
// accelerometer, magnetometer, and gyroscope over time and prints formatted
// data over serial.
//
// Data acquisition starts after 40 * P, where P is the period in seconds. This
// is so there is sufficient data points to filter out gravity. Feel free to
// change the constant STABLE_ITERS to suit your needs.
// 
// Serial input state machine is used to begin data acquisition. Input will be
// in the form N[int]P[float]G, where the numbers after N is an integer
// representing the number of data points to collect, and P is the period at
// which to sample them, in seconds (P = 1/freq). A 'G' must be sent to begin 
// the transmission of data over serial.
// 
// Hexiwear will wait for serial input to come, and then begin data acquisition.
// Program can be modified so that serial output begins right away, just remove
// state machine code.
//
// This program was designed to be used with Python script get_data.py.
////////////////////////////////////////////////////////////////////////////////


#include "mbed.h"
#include "FXOS8700.h"
#include "FXAS21002.h"

#include <ctime>

#define STABLE_ITERS 40

DigitalOut led1(LED_GREEN);

// Initialize Serial port
Serial pc(USBTX, USBRX, 115200);

// Pin connections & address for Hexiwear
FXOS8700 accel(PTC11, PTC10);
FXOS8700 mag(PTC11, PTC10);
FXAS21002 gyro(PTC11,PTC10);

// main() runs in its own thread in the OS
// (note the calls to Thread::wait below for delays)
int main() {

    // variable definitions
    int i = 0;

    // clock variables
    std::clock_t start = std::clock(), mid = std::clock();
    float duration;

    // arrays used to retrieve sensor data
    float accel_data[3];
    float mag_data[3];
    float gyro_data[3];

    // gravity low-pass filter
    // https://developer.android.com/reference/android/hardware/SensorEvent.html#values
    float gravity[3] = {0.0, 0.0, 0.0};
    float alpha = 0.8;

    // variables to read over serial
    char c;
    int num_iters = 0;
    float period = 0;
    bool READ_ITERS = true;
    float MULTIPLIER = 1.0;

    // Configure Accelerometer FXOS8700, Magnetometer FXOS8700, Gyroscope FXAS21002
    accel.accel_config();
    mag.mag_config();
    gyro.gyro_config();

    // infinite loop
    while(true) {
        ////////////////////////////////////////////////////////////////////////
        // SERIAL PART: very simple state machine that receives characters over
        //              serial in the form N[int]P[float]G, where the numbers
        //              after N is an integer representing the number of data
        //              points to collect, and P is the period at which to
        //              sample them (P = 1/freq). A 'G' must be sent to begin
        //              the transmission of data over serial.
        ////////////////////////////////////////////////////////////////////////
        // block until we get a character
        c = pc.getc();

        if(c == 'N') {
            // change number of iterations
            READ_ITERS = true;
            MULTIPLIER = 1;
        } else if(c == 'P') {
            // change period
            READ_ITERS = false;
            MULTIPLIER = 1;
        } else if(c == '.') {
            // got decimal, presumably for period. Change multiplier
            MULTIPLIER = 0.1;
        } else if(c >= '0' && c <= '9') {
            // got number, change corresponding variable
            if(READ_ITERS) {
                num_iters = 10 * num_iters + (int)(c-'0');
            } else {
                // depending on if we're before or after decimal, change period
                if(MULTIPLIER >= 0.9) {
                    period = 10 * period + (int)(c-'0');
                } else {
                    period = period + (float)(c-'0') * MULTIPLIER;
                    MULTIPLIER /= 10.0;
                }
            }
        } else if(c == 'G') {
            // received a G, so we should start collecting data
            i = 0;
            // printf("NUM_ITERS=%d, PERIOD=%f\n", num_iters, period);
            printf("Begin Data Acquisition from FXOS8700CQ and FXAS21002 sensors....\r\n\r\n");
            wait(0.1);
            
            // loop until gravity is stable (low-pass), then start collecting data
            while (i < num_iters + STABLE_ITERS) {
                mid = std::clock();
              
                // get data
                accel.acquire_accel_data_g(accel_data);
                mag.acquire_mag_data_uT(mag_data);
                gyro.acquire_gyro_data_dps(gyro_data);

                // low-pass filter for gravity (subtract gravity from accel_data)
                gravity[0] = alpha * gravity[0] + (1 - alpha) * accel_data[0];
                gravity[1] = alpha * gravity[1] + (1 - alpha) * accel_data[1];
                gravity[2] = alpha * gravity[2] + (1 - alpha) * accel_data[2];

                // data acquisition actually starts now
                if(i == STABLE_ITERS) {
                    // send NOW to show where data transmission begins
                    printf("NOW.\n");
                    led1 = !led1;
                    // used to debug if we're sampling for the correct duration
                    start = std::clock();
                }

                // have to wait for low-pass filter to normalize before filtering gravity
                if(i >= STABLE_ITERS) {
                    // print to serial
                    printf("%d//acc x:%f,y:%f,z:%f;mag x:%f,y:%f,z:%f;gyr r:%f,p:%f,y:%f\r\n",
                           i-40,
                           accel_data[0]-gravity[0], accel_data[1]-gravity[1], accel_data[2]-gravity[2],
                           mag_data[0], mag_data[1], mag_data[2], 
                           gyro_data[1], gyro_data[1], gyro_data[2]);
                }

                i++;

                // compute how long computations and printf took and subtract it from wait
                duration = (std::clock() - mid) / (float) CLOCKS_PER_SEC;
                // only wait if computation time is less than wait time
                if(duration < period) {
                    wait(period - duration);
                }
            }
            // send STOP to show where data transmission ends
            printf("STOP.\n");

            duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;

            printf("Expected time elapsed: %f.\r\n", num_iters * period);
            printf("Data Acquisition done. Time elapsed: %f.\r\n", duration);

            printf("END\r\n");

            // reset num_iters and period
            num_iters = 0;
            period = 0;
        }
    }
    

    return 0;
}