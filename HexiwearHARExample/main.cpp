#include "models/dnn6_2.hpp"  //gernerated model file
#include "tensor.hpp"  //useful tensor classes
#include "mbed.h"
#include "FXOS8700.h"
#include "FXAS21002.h"

#include <ctime>

//  Check out the full featured example application for interfacing to the 
//  Accelerometer/Magnetometer device at the following URL
//  https://developer.mbed.org/users/trm/code/fxos8700cq_example/

//DigitalOut led1(LED_GREEN);

// Initialize Serial port
Serial pc(USBTX, USBRX, 115200);

// Pin connections & address for Hexiwear
FXOS8700 accel(PTC11, PTC10);
//FXOS8700 mag(PTC11, PTC10);
FXAS21002 gyro(PTC11,PTC10);

// main() runs in its own thread in the OS
// (note the calls to Thread::wait below for delays)
int main() {

	while(1) {

	    int i = 0;

	    // clock variables
	    std::clock_t mid = std::clock();
	    float duration;
	    
	    // Configure Accelerometer FXOS8700, Magnetometer FXOS8700
	    accel.accel_config();
	    //mag.mag_config();
	    gyro.gyro_config();

	    string LABELS[] = {"WALKING",
	                       "WALKING_UPSTAIRS",
	                       "WALKING_DOWNSTAIRS",
	                       "SITTING",
	                       "STANDING",
	                       "LAYING"};

	    float accel_data[3];
	    float gyro_data[3];

	    float gravity[3] = {0.0, 0.0, 0.0};
	    float alpha = 0.8;
	    
	    float input_data[128*6];

	    printf("DNN_BEGIN: Begin Data Acquisition from sensors....\r\n\r\n");
	    wait(0.1);
	    //printf("DEBUG1\r\n");
	    
	    while (i < 200) {
	        mid = std::clock();
	        accel.acquire_accel_data_g(accel_data);
	        gyro.acquire_gyro_data_dps(gyro_data);

	        gravity[0] = alpha * gravity[0] + (1 - alpha) * accel_data[0];
	        gravity[1] = alpha * gravity[1] + (1 - alpha) * accel_data[1];
	        gravity[2] = alpha * gravity[2] + (1 - alpha) * accel_data[2];

	        if(i == 72) {
	            //printf("Now.\n\n");
	        }

	        if(i >= 72) {
	            input_data[(i-72)*6+0] = (accel_data[0] - gravity[0]);
	            input_data[(i-72)*6+1] = (accel_data[1] - gravity[1]);
	            input_data[(i-72)*6+2] = (accel_data[2] - gravity[2]);

	            input_data[(i-72)*6+3] = gyro_data[0]/100.0;
	            input_data[(i-72)*6+4] = gyro_data[1]/100.0;
	            input_data[(i-72)*6+5] = gyro_data[2]/100.0;

	            printf("%d: x %f y %f z %f r %f p %f y %f\n",i,
	                   accel_data[0] - gravity[0],accel_data[1] - gravity[1],accel_data[2] - gravity[2],
	                   gyro_data[0],gyro_data[1],gyro_data[2]);
	        }

	        // compute how long computations and printf took and subtract it from wait
	        duration = (std::clock() - mid) / (float) CLOCKS_PER_SEC;
	        // only wait if computation time is less than wait time
	        if(duration < 0.02) {
	            wait(0.02 - duration);
	        }

	        i++;
	    }
	    

	    printf("Data Acquisition done.\n\n");
	    
		/*
	    for(i = 0; i < 128; i++) {
	        printf("%d: x %f y %f z %f r %f p %f y %f\n",i,
	               input_data[i*6+0],input_data[i*6+1],input_data[i*6+2],
	               input_data[i*6+3],input_data[i*6+4],input_data[i*6+5]);
	    }
	    */

	    Context ctx;  //creating the context class, the stage where inferences take place 
	    // wrapping the input data in a tensor class
	    Tensor* input_x = new WrappedRamTensor<float>({1, 128*6}, (float*) input_data);

	    get_dnn6_2_ctx(ctx, input_x);  // pass the tensor to the context
	    S_TENSOR pred_tensor = ctx.get("y_pred:0");  // getting a reference to the output tensor
	    ctx.eval(); //trigger the inference

	    int pred_label = *(pred_tensor->read<int>(0, 0));  //getting the result back
	    printf("Predicted label: ");
	    printf(LABELS[pred_label].c_str());
	    printf("\n\n");

	    wait(0.2);
	}

    return 0;
}