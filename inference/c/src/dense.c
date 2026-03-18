void dense(
    float *input,    // [in_features]
    float *weights,  // [out_features * in_features]
    float *bias,     // [out_features]
    float *output,   // [out_features]
    int in_features,
    int out_features
) {
    for (int o = 0; o < out_features; o++) {

        float sum = bias[o];

        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weights[o * in_features + i];
        }

        output[o] = sum;
    }
}