#include <stdio.h>
#include <stdint.h>

void convertRgb2Gray_host(uchar3 * rgbPic, int width, int height, uint8_t * grayPic) {
    for (int r = 0; r < height; ++r) 
        for (int c = 0; c < width; ++c) {
            int i = r * width + c;
            grayPic[i] = 0.299f * rgbPic[i].x + 0.587f * rgbPic[i].y + 0.114f * rgbPic[i].z;
        }
}