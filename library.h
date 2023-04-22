#ifndef HEADER_H
#define HEADER_H

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

/**
 * Working with files
 * File types:
 * - .pnm file with "P2" or "P3" header
 * - filter (plain text)
 */
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 *pixels, int width, int height, int originalWidth, char *fileName);

/**
 * convert RGB to Grayscale
 */
void convertRgb2Gray_host(uchar3 * rgbPic, int width, int height, uint8_t * grayPic);

/**
 * Another functions
 */
void printDeviceInfo();
void printError(char * msg, uchar3 * in1, uchar3 * in2, int width, int height);
char * concatStr(const char * s1, const char * s2);

/**
 * Time counter
 */
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, 0);                                                                 
        cudaEventSynchronize(start);
    }

    void Stop() {
        cudaEventRecord(stop, 0);
    }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }

    void printTime(char * s) {
        printf("Processing time of %s: %f ms\n\n", s, Elapsed());
    }
};

#endif