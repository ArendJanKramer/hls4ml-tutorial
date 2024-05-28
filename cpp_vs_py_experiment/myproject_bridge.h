#ifndef MYPROJECT_BRIDGE_H_
#define MYPROJECT_BRIDGE_H_

#include "firmware/myproject.h"

// hls-fpga-machine-learning insert bram

extern "C" {

class MyprojectClassFloat_344c2d52;

// Wrapper of top level function for Python bridge
void myproject_float(
    float fc1_input[N_INPUT_1_1],
    float layer13_out[N_LAYER_11]
    );

class MyprojectClassDouble_344c2d52;

void myproject_double(
    double fc1_input[N_INPUT_1_1],
    double layer13_out[N_LAYER_11]
    );
}

#endif
