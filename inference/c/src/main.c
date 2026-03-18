#include <stdio.h>
#include "model.h"
#include "config.h"

int main() {
    float input[INPUT_LEN] = {0};  // TODO: load real ECG
    float output[NUM_CLASSES];

    forward(input, output);

    printf("Output:\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}