#include "config.h"
#include "model.h"

// Forward declarations
void conv1d(float*, float*, float*, float*, int, int, int, int);
void relu(float*, int);
void dense(float*, float*, float*, float*, int, int);

// TEMP: dummy weights (replace later)
static float conv1_w[C1_OUT * IN_CH * C1_K] = {0};
static float conv1_b[C1_OUT] = {0};

static float conv2_w[C2_OUT * C1_OUT * C2_K] = {0};
static float conv2_b[C2_OUT] = {0};

static float fc_w[NUM_CLASSES * C2_OUT] = {0};
static float fc_b[NUM_CLASSES] = {0};

void forward(float *input, float *output) {

    int len1 = INPUT_LEN - C1_K + 1;
    int len2 = len1 - C2_K + 1;

    static float conv1_out[C1_OUT * (INPUT_LEN - C1_K + 1)];
    static float conv2_out[C2_OUT * (INPUT_LEN - C1_K - C2_K + 2)];
    static float gap_out[C2_OUT];

    // --- Conv1 ---
    conv1d(input, conv1_w, conv1_b, conv1_out,
           IN_CH, C1_OUT, INPUT_LEN, C1_K);

    relu(conv1_out, C1_OUT * len1);

    // --- Conv2 ---
    conv1d(conv1_out, conv2_w, conv2_b, conv2_out,
           C1_OUT, C2_OUT, len1, C2_K);

    relu(conv2_out, C2_OUT * len2);

    // --- Global Average Pool ---
    for (int c = 0; c < C2_OUT; c++) {
        float sum = 0.0f;
        for (int i = 0; i < len2; i++) {
            sum += conv2_out[c * len2 + i];
        }
        gap_out[c] = sum / len2;
    }

    // --- Dense ---
    dense(gap_out, fc_w, fc_b, output, C2_OUT, NUM_CLASSES);
}