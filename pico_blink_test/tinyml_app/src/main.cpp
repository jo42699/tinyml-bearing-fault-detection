
#include <cstdio>
#include <cstdint>

#include "pico/stdlib.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "bearing_model.h"

// Model configuration
constexpr int INPUT_SIZE = 1024;
constexpr int NUM_CLASSES = 4;

// Initial tensor arena: 128 KB
alignas(16) static uint8_t tensor_arena[128 * 1024];

const char* class_names[NUM_CLASSES] = {
    "Healthy",
    "Inner race fault",
    "Ball fault",
    "Outer race fault"
};

int main() {
    stdio_init_all();
    sleep_ms(2000);

    printf("\n============================\n");
    printf("Bearing TinyML - Pico 2 ARM\n");
    printf("============================\n");

    // Load model
    const tflite::Model* model = tflite::GetModel(bearing_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model schema version mismatch\n");
        while (1) sleep_ms(1000);
    }

    // Register the seven required operators
    tflite::MicroMutableOpResolver<7> resolver;

    if (resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk) {
        printf("ERROR: Operator registration failed\n");
        while (1) sleep_ms(1000);
    }

    // Create interpreter
    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        sizeof(tensor_arena)
    );

    // Allocate memory
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Tensor allocation failed\n");
        while (1) sleep_ms(1000);
    }

    printf("Tensor allocation successful\n");
    printf("Arena used: %u bytes\n",
           static_cast<unsigned>(interpreter.arena_used_bytes()));

    // Get input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input == nullptr || output == nullptr ||
        input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8 ||
        input->bytes != INPUT_SIZE ||
        output->bytes != NUM_CLASSES) {
        printf("ERROR: Unexpected tensor configuration\n");
        while (1) sleep_ms(1000);
    }

    printf("Input tensor: INT8, %d bytes\n", INPUT_SIZE);
    printf("Output tensor: INT8, %d classes\n", NUM_CLASSES);

    // Test with a zero-valued vibration window.
    // Quantized zero is represented by the input zero point.
    for (int i = 0; i < INPUT_SIZE; i++) {
        input->data.int8[i] =
            static_cast<int8_t>(input->params.zero_point);
    }

    // Run inference
    if (interpreter.Invoke() != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        while (1) sleep_ms(1000);
    }

    // Find highest-scoring class
    int best_class = 0;

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output->data.int8[i] > output->data.int8[best_class]) {
            best_class = i;
        }
    }

    printf("\nInference successful!\n");
    printf("Predicted class: %s\n", class_names[best_class]);

    printf("Output scores:\n");

    for (int i = 0; i < NUM_CLASSES; i++) {
        float score =
            (output->data.int8[i] - output->params.zero_point)
            * output->params.scale;

        printf("  %s: %.4f\n", class_names[i], score);
    }

    printf("\nInference test complete.\n");

    while (1) {
        sleep_ms(1000);
    }
}
