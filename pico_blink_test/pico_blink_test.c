
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/gpio.h"

// Inference function implemented in inference.cpp
int run_bearing_inference(void);

int main()
{
    // Initialise USB serial
    stdio_init_all();

    printf("Waiting for USB Serial Monitor...\n");
    fflush(stdout);

    // Wait until COM port is connected
    while (!stdio_usb_connected())
    {
        sleep_ms(100);
    }

    sleep_ms(500);

    printf("\n");
    printf("====================================\n");
    printf("   Pico 2 TinyML Bearing Detection  \n");
    printf("====================================\n");
    fflush(stdout);

    // Bearing fault classes
    const char* class_names[] = {
        "Healthy",
        "Inner Race",
        "Ball Fault",
        "Outer Race"
    };

    // Start inference
    printf("Starting bearing inference...\n");
    fflush(stdout);

    printf("BEFORE INFERENCE\n");
    fflush(stdout);

    // Run TensorFlow Lite Micro model
    int result = run_bearing_inference();

    printf("AFTER INFERENCE\n");
    fflush(stdout);

    printf("Inference returned: %d\n", result);

    // Display prediction
    if (result >= 0 && result < 4)
    {
        printf("Predicted bearing condition: %s\n",
               class_names[result]);
    }
    else
    {
        printf("Bearing inference failed.\n");
    }

    printf("Inference test completed.\n");
    fflush(stdout);

    // Initialise onboard LED
    const uint LED_PIN = 25;

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    printf("Starting LED heartbeat...\n");
    fflush(stdout);

    // Blink LED and print heartbeat
    while (1)
    {
        printf("Pico is alive - LED ON\n");
        fflush(stdout);

        gpio_put(LED_PIN, 1);
        sleep_ms(1000);

        printf("Pico is alive - LED OFF\n");
        fflush(stdout);

        gpio_put(LED_PIN, 0);
        sleep_ms(1000);
    }

    return 0;
}



