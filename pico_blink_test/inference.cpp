
#include <cstdio>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model/bearing_model.h"
#include "model/test_input.h"

// 128 KB tensor arena
alignas(16) static uint8_t tensor_arena[128 * 1024];

extern "C" int run_bearing_inference(void)
{
    printf("\n>>> INFERENCE FUNCTION ENTERED <<<\n");
    fflush(stdout);

    // Load model
    printf("Loading model...\n");
    fflush(stdout);

    const tflite::Model* model = tflite::GetModel(bearing_model);

    if (model == nullptr)
    {
        printf("ERROR: Model loading failed\n");
        fflush(stdout);
        return -1;
    }

    printf("Model loaded successfully.\n");

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("ERROR: Model schema version mismatch\n");
        fflush(stdout);
        return -1;
    }

    printf("Model schema verified.\n");

    // Register required operators
    printf("Registering operators...\n");
    fflush(stdout);

    tflite::MicroMutableOpResolver<7> resolver;

    if (resolver.AddExpandDims() != kTfLiteOk ||
        resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddMaxPool2D() != kTfLiteOk ||
        resolver.AddMean() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk)
    {
        printf("ERROR: Operator registration failed\n");
        fflush(stdout);
        return -1;
    }

    printf("Operators registered successfully.\n");

    // Create interpreter
    printf("Creating interpreter...\n");
    fflush(stdout);

    tflite::MicroInterpreter interpreter(
        model,
        resolver,
        tensor_arena,
        sizeof(tensor_arena)
    );

    // Allocate tensors
    printf("Allocating tensor memory...\n");
    fflush(stdout);

    if (interpreter.AllocateTensors() != kTfLiteOk)
    {
        printf("ERROR: Tensor allocation failed\n");
        fflush(stdout);
        return -1;
    }

    printf("Tensor allocation successful.\n");

    // Get input tensor
    TfLiteTensor* input = interpreter.input_tensor(0);

    if (input == nullptr ||
        input->type != kTfLiteInt8 ||
        input->dims->size != 3 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != 1024 ||
        input->dims->data[2] != 1)
    {
        printf("ERROR: Input tensor mismatch\n");
        fflush(stdout);
        return -1;
    }

    printf("Input tensor verified.\n");

    // Copy input vibration samples
    memcpy(input->data.int8, test_input, 1024);

    printf("Input data copied.\n");
    fflush(stdout);

    // Run inference
    printf("Invoking model...\n");
    fflush(stdout);

    if (interpreter.Invoke() != kTfLiteOk)
    {
        printf("ERROR: Inference execution failed\n");
        fflush(stdout);
        return -1;
    }

    printf("Model invocation successful.\n");

    // Get output tensor
    TfLiteTensor* output = interpreter.output_tensor(0);

    if (output == nullptr ||
        output->type != kTfLiteInt8 ||
        output->dims->size != 2 ||
        output->dims->data[0] != 1 ||
        output->dims->data[1] != 4)
    {
        printf("ERROR: Output tensor mismatch\n");
        fflush(stdout);
        return -1;
    }

    // Calculate predicted class
    int predicted_class = 0;
    int8_t highest_score = output->data.int8[0];

    const char* class_names[] = {
        "Healthy",
        "Inner Race",
        "Ball Fault",
        "Outer Race"
    };

    printf("\nClass scores:\n");

    for (int i = 0; i < 4; i++)
    {
        int8_t score = output->data.int8[i];

        float confidence =
            (score - output->params.zero_point) *
            output->params.scale;

        printf("%s: %.2f%%\n",
               class_names[i],
               confidence * 100.0f);

        if (score > highest_score)
        {
            highest_score = score;
            predicted_class = i;
        }
    }

    printf("\nInference completed successfully!\n");
    fflush(stdout);

    return predicted_class;
}
