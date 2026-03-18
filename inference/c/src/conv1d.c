#include "config.h"

void conv1d(
    float *input,        // [in_ch * input_len]
    float *weights,      // [out_ch * in_ch * kernel]
    float *bias,         // [out_ch]
    float *output,       // [out_ch * output_len]
    int in_ch,
    int out_ch,
    int input_len,
    int kernel
) {
    int output_len = input_len - kernel + 1;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int i = 0; i < output_len; i++) {

            float sum = bias[oc];

            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {

                    int in_idx = ic * input_len + (i + k);
                    int w_idx = oc * (in_ch * kernel) + ic * kernel + k;

                    sum += input[in_idx] * weights[w_idx];
                }
            }

            output[oc * output_len + i] = sum;
        }
    }
}