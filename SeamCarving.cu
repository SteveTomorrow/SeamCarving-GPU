#include <stdio.h>
#include <stdint.h>
#include "./src/library.h"
using namespace std;

// Seam Carving cu C++ GPU

int WIDTH;
__device__ int d_WIDTH;

int xSobel[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
int ySobel[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
__constant__ int d_xSobel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__constant__ int d_ySobel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
const int filterWidth = 3;
const int blockSize = 32;




/**
 * @param argc[1] name of the input file (.pmn)
 * @param argc[2] name of output file with no extension, created by using host & device
 * @param argc[3] horizontal of image you want to resize 
 * @param argc[4] - optional - default(32): blocksize.x
 * @param argc[5] - optional - default(32): blocksize.y
 */
void checkInput(int argc, char ** argv, int &width, int &height, uchar3 * &rgbPic, int &desiredWidth, dim3 &blockSize) {
    if (argc != 4 && argc != 6) {
        printf("The number of arguments is invalid\n");
        exit(EXIT_FAILURE);
    }

    // Read file
    readPnm(argv[1], width, height, rgbPic);
    printf("Image size (width x height): %i x %i\n\n", width, height);

    WIDTH = width;
    CHECK(cudaMemcpyToSymbol(d_WIDTH, &width, sizeof(int)));
    // Check user's desired width
    desiredWidth = atoi(argv[3]);

    if (desiredWidth <= 0 || desiredWidth >= width) {
        printf("Your desired width must between 0 & current picture's width!\n");
        exit(EXIT_FAILURE);
    }

    // Block size
    if (argc == 6) {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    } 


    // Check GPU is working or not
    printDeviceInfo();
}


__global__ void convertRgb2GrayKernel(uchar3 * rgbPic, int width, int height, uint8_t * grayPic) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < height && c < width) {
        int i = r * width + c;
        grayPic[i] = 0.299f*rgbPic[i].x + 0.587f*rgbPic[i].y + 0.114f*rgbPic[i].z;
    }
}


__global__ void calEnergy(uint8_t * inPixels, int width, int height, int * energy) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int s_width = blockDim.x + filterWidth - 1;
    int s_height = blockDim.y + filterWidth - 1;

    // Each block loads data from GMEM to SMEM
    extern __shared__ uint8_t s_inPixels[];

    int readRow = row - filterWidth / 2, readCol, tmpRow, tmpCol;
    int firstReadCol = col - filterWidth / 2;
    int virtualRow, virtualCol;

    for (virtualRow = threadIdx.y; virtualRow < s_height; readRow += blockDim.y, virtualRow += blockDim.y) {
        tmpRow = readRow;

        // if (readRow < 0)
        //     readRow = 0;
        // else if (readRow >= height) 
        //     readRow = height - 1;

        readRow = min(max(readRow, 0), height - 1);//0 <= readCol <= height-1
        
        readCol = firstReadCol;
        virtualCol = threadIdx.x;

        for (; virtualCol < s_width; readCol += blockDim.x, virtualCol += blockDim.x) {
            tmpCol = readCol;

            // if (readCol < 0) 
            //     readCol = 0;
            // else if (readCol >= width) 
            //     readCol = width - 1;

            readCol = min(max(readCol, 0), width - 1);// 0 <= readCol <= width-1
            
            s_inPixels[virtualRow * s_width + virtualCol] = inPixels[readRow * d_WIDTH + readCol];
            readCol = tmpCol;
        }
        readRow = tmpRow;
    } 
    __syncthreads();


    // Each thread compute energy on SMEM
    int x_kernel = 0, y_kernel = 0;
    for (int i = 0; i < filterWidth; ++i) {
        for (int j = 0; j < filterWidth; ++j) {
            uint8_t closest = s_inPixels[(threadIdx.y + i) * s_width + threadIdx.x + j];
            int filterIdx = i * filterWidth + j;
            x_kernel += closest * d_xSobel[filterIdx];
            y_kernel += closest * d_ySobel[filterIdx];
        }
    }

    // Each thread writes result from SMEM to GMEM
    if (col < width && row < height)
        energy[row * d_WIDTH + col] = abs(x_kernel) + abs(y_kernel);
}
__global__ void calEnergy2(uint8_t *inPixels, int width, int height, int *energy) {
    // Define the size of the tile and the overlap region
    constexpr int TILE_WIDTH = 32;
    constexpr int TILE_HEIGHT = 8;
    constexpr int TILE_OVERLAP = filterWidth / 2;

    // Define the shared memory buffer
    __shared__ uint8_t s_inPixels[(TILE_HEIGHT + filterWidth - 1) * (TILE_WIDTH + filterWidth - 1)];

    // Calculate the row and column indices
    const int globalRow = blockIdx.y * (TILE_HEIGHT - TILE_OVERLAP * 2) + threadIdx.y - TILE_OVERLAP;
    const int globalCol = blockIdx.x * (TILE_WIDTH - TILE_OVERLAP * 2) + threadIdx.x - TILE_OVERLAP;
    const int threadIndex = threadIdx.y * TILE_WIDTH + threadIdx.x;
    const int tileWidth = TILE_WIDTH + filterWidth - 1;
    const int tileHeight = TILE_HEIGHT + filterWidth - 1;

    // Load data from GMEM to SMEM
    #pragma unroll
    for (int i = 0; i < tileHeight; i += TILE_HEIGHT) {
        #pragma unroll
        for (int j = 0; j < tileWidth; j += TILE_WIDTH) {
            // Calculate the indices for the tile
            const int tileRow = globalRow + i;
            const int tileCol = globalCol + j;
            const int tileIndex = (i + threadIdx.y) * tileWidth + (j + threadIdx.x);

            // Load data into the tile
            if (tileRow >= 0 && tileRow < height && tileCol >= 0 && tileCol < width) {
                s_inPixels[tileIndex] = inPixels[tileRow * width + tileCol];
            } else {
                s_inPixels[tileIndex] = 0;
            }
        }
    }

    // Synchronize threads to ensure data is fully loaded
    __syncthreads();

    // Calculate energy for the current thread
    int x_kernel = 0, y_kernel = 0;
    #pragma unroll
    for (int i = 0; i < filterWidth; ++i) {
        #pragma unroll
        for (int j = 0; j < filterWidth; j += 4) {
            const int closest1 = s_inPixels[(threadIdx.y + i) * tileWidth + threadIdx.x + j];
            const int filterIdx = i * filterWidth + j;
            x_kernel += closest1 * d_xSobel[filterIdx];
            y_kernel += closest1 * d_ySobel[filterIdx];
        }
    }

    // Each thread writes result from SMEM to GMEM
    const int globalIndex = globalRow * width + globalCol;
    if (globalRow >= 0 && globalRow < height && globalCol >= 0 && globalCol < width) {
        energy[globalIndex] = abs(x_kernel) + abs(y_kernel);
    }
}

__global__ void carvingKernel(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    int row = blockIdx.x;
    int baseIdx = row * d_WIDTH;
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        energy[baseIdx + i] = energy[baseIdx + i + 1];
    }
}

__global__ void carvingKernel2(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    __shared__ uchar3 sharedPixels[blockSize]; // shared memory for output pixels
    __shared__ uint8_t sharedGrays[blockSize]; // shared memory for gray pixels
    __shared__ int sharedEnergy[blockSize]; // shared memory for energy

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int baseIdx = row * width;
    int idx = baseIdx + tid;

    // Load data into shared memory
    if (tid < width) {
        sharedPixels[tid] = outPixels[idx];
        sharedGrays[tid] = grayPixels[idx];
        sharedEnergy[tid] = energy[idx];
    }

    __syncthreads(); // Ensure all threads have loaded data into shared memory

    // Update output pixels, gray pixels, and energy
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        if (tid < width - 1) {
            sharedPixels[tid] = (tid < i) ? sharedPixels[tid] : sharedPixels[tid + 1];
            sharedGrays[tid] = (tid < i) ? sharedGrays[tid] : sharedGrays[tid + 1];
            sharedEnergy[tid] = (tid < i) ? sharedEnergy[tid] : sharedEnergy[tid + 1];
        }
        __syncthreads(); // Ensure all threads have updated data in shared memory
    }

    // Write data back to global memory
    if (tid < width) {
        outPixels[idx] = sharedPixels[tid];
        grayPixels[idx] = sharedGrays[tid];
        energy[idx] = sharedEnergy[tid];
    }
}


_global_ void carvingKernel3(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    _shared_ uchar3 sharedOutPixels[16];
    _shared_ uint8_t sharedGrayPixels[16];
    _shared_ int sharedEnergy[16];

    int row = blockIdx.x;
    int baseIdx = row * d_WIDTH;
    int leastSignificant = leastSignificantPixel[row];


    for (int i = leastSignificant + threadIdx.x; i < width - 1; i += blockDim.x) {
        int idx = baseIdx + i;

        // Copy a row of data into shared memory
        sharedOutPixels[threadIdx.x] = outPixels[idx + 1];
        sharedGrayPixels[threadIdx.x] = grayPixels[idx + 1];
        sharedEnergy[threadIdx.x] = energy[idx + 1];

        __syncthreads();

        // Compute values for the current row using the shared data
        if (i < width - 1) {
            outPixels[idx] = sharedOutPixels[threadIdx.x];
            grayPixels[idx] = sharedGrayPixels[threadIdx.x];
            energy[idx] = sharedEnergy[threadIdx.x];
        }

        __syncthreads();
    }
}

__global__ void findSeamKernel(int * minimalEnergy, int *leastSignificantPixel, int width, int height) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= height) return;

    int minCol = 0, c = width - 1;
    for (int j = 1; j < width; j++) {
        if (minimalEnergy[r * width + j] < minimalEnergy[r * width + minCol]) {
            minCol = j;
        }
    }

    for (; c >= 0; --c) {
        leastSignificantPixel[r * width + c] = minCol;
        if (r > 0) {
            int aboveIdx = (r - 1) * width + c;
            int min = minimalEnergy[aboveIdx], minColCpy = minCol;

            if (minColCpy > 0 && minimalEnergy[aboveIdx - 1] < min) {
                min = minimalEnergy[aboveIdx - 1];
                minCol = minColCpy - 1;
            }
            if (minColCpy < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
        }
        __syncthreads();
    }
}


__global__ void energyToTheEndKernel(int * energy, int * minimalEnergy, int width, int height, int fromRow) {
    size_t halfBlock = blockDim.x / 2;//blockDim.x >> 1

    int col = blockIdx.x * halfBlock - halfBlock + threadIdx.x;

    if (fromRow == 0 && col >= 0 && col < width) {
        minimalEnergy[col] = energy[col];
    }
    __syncthreads();

    for (int stride = fromRow != 0 ? 0 : 1; stride < halfBlock && fromRow + stride < height; ++stride) {
        if (threadIdx.x < blockDim.x - (stride << 1)) {
            int curRow = fromRow + stride;
            int curCol = col + stride;

            if (curCol >= 0 && curCol < width) {
                int idx = curRow * d_WIDTH + curCol;
                int aboveIdx = (curRow - 1) * d_WIDTH + curCol;

                int min = minimalEnergy[aboveIdx];
                if (curCol > 0 && minimalEnergy[aboveIdx - 1] < min)
                    min = minimalEnergy[aboveIdx - 1];
                
                if (curCol < width - 1 && minimalEnergy[aboveIdx + 1] < min)
                    min = minimalEnergy[aboveIdx + 1];
                

                minimalEnergy[idx] = min + energy[idx];
            }
        }
        __syncthreads();
    }
}


void findSeam(int * minimalEnergy, int *leastSignificantPixel, int width, int height) {
    int minCol = 0, r = height - 1;

    for (int c = 1; c < width; ++c)
        if (minimalEnergy[r * WIDTH + c] < minimalEnergy[r * WIDTH + minCol])
            minCol = c;
    
    for (; r >= 0; --r) {
        leastSignificantPixel[r] = minCol;
        if (r > 0) {
            int aboveIdx = (r - 1) * WIDTH + minCol;
            int min = minimalEnergy[aboveIdx], minColCpy = minCol;

            if (minColCpy > 0 && minimalEnergy[aboveIdx - 1] < min) {
                min = minimalEnergy[aboveIdx - 1];
                minCol = minColCpy - 1;
            }
            if (minColCpy < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                minCol = minColCpy + 1;
            }
        }
    }
}


// HOST

int getPixelEnergy(uint8_t * grayPixels, int row, int col, int width, int height) {
    int x_kernel = 0;
    int y_kernel = 0;

    for (int i = 0; i < 3; ++i) { // 3: filter width
        for (int j = 0; j < 3; ++j) {
            int r = min(max(0, row - 1 + i), height - 1); // 0 <= row - 1 + i < height
            int c = min(max(0, col - 1 + j), width - 1); // 0 <= col - 1 + j < width

            uint8_t pixelVal = grayPixels[r * WIDTH + c];//

            x_kernel += pixelVal * xSobel[i][j];// Convolution with x-Sobel
            y_kernel += pixelVal * ySobel[i][j];// Convolution with y-Sobel
        }
    }
    return abs(x_kernel) + abs(y_kernel);// Add matrix
}



void energyToTheEnd(int * energy, int * minimalEnergy, int width, int height) {
    for (int c = 0; c < width; ++c) {
        minimalEnergy[c] = energy[c];
    }
    for (int r = 1; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = r * WIDTH + c;
            int aboveIdx = (r - 1) * WIDTH + c;

            int min = minimalEnergy[aboveIdx];
            if (c > 0 && minimalEnergy[aboveIdx - 1] < min) {
                min = minimalEnergy[aboveIdx - 1];
            }
            if (c < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                min = minimalEnergy[aboveIdx + 1];
            }

            minimalEnergy[idx] = min + energy[idx];
        }
    }
}

void hostResizing(uchar3 * inPixels, int width, int height, int desiredWidth, uchar3 * outPixels) {
    GpuTimer timer;
    timer.Start();

    memcpy(outPixels, inPixels, width * height * sizeof(uchar3));

    // Allocating memory
    int * energy = (int *)malloc(width * height * sizeof(int));
    int * minimalEnergy = (int *)malloc(width * height * sizeof(int));
    
    // Get grayscale
    uint8_t * grayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray_host(inPixels, width, height, grayPixels);

    // Calculate all pixels energy
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            energy[r * WIDTH + c] = getPixelEnergy(grayPixels, r, c, width, height);
        }
    }

    while (width > desiredWidth) {
        // Calculate energy to the end. (go from bottom to top)
        energyToTheEnd(energy, minimalEnergy, width, height);

        // find min index of last row
        int minCol = 0, r = height - 1, prevMinCol;
        for (int c = 1; c < width; ++c) {
            if (minimalEnergy[r * WIDTH + c] < minimalEnergy[r * WIDTH + minCol])
                minCol = c;
        }

        // Find and remove seam from last to first row
        for (; r >= 0; --r) {
            // remove seam pixel on row r
            for (int i = minCol; i < width - 1; ++i) {
                outPixels[r * WIDTH + i] = outPixels[r * WIDTH + i + 1];
                grayPixels[r * WIDTH + i] = grayPixels[r * WIDTH + i + 1];
                energy[r * WIDTH + i] = energy[r * WIDTH + i + 1];
            }

            // Update energy
            if (r < height - 1) {
                int affectedCol = max(0, prevMinCol - 2);

                while (affectedCol <= prevMinCol + 2 && affectedCol < width - 1) {
                    energy[(r + 1) * WIDTH + affectedCol] = getPixelEnergy(grayPixels, r + 1, affectedCol, width - 1, height);
                    affectedCol += 1;
                }
            }

            // find to the top
            if (r > 0) {
                prevMinCol = minCol;

                int aboveIdx = (r - 1) * WIDTH + minCol;
                int min = minimalEnergy[aboveIdx], minColCpy = minCol;
                if (minColCpy > 0 && minimalEnergy[aboveIdx - 1] < min) {
                    min = minimalEnergy[aboveIdx - 1];
                    minCol = minColCpy - 1;
                }
                if (minColCpy < width - 1 && minimalEnergy[aboveIdx + 1] < min) {
                    minCol = minColCpy + 1;
                }
            }
        }

        int affectedCol;
        for (affectedCol=max(0, minCol - 2); affectedCol <= minCol + 2 && affectedCol < width - 1; ++affectedCol) {
            energy[affectedCol] = getPixelEnergy(grayPixels, 0, affectedCol, width - 1, height);
        }

        --width;
    }
    
    free(grayPixels);
    free(minimalEnergy);
    free(energy);

    timer.Stop();
    timer.printTime((char *)"host");
}

//device

void deviceResizing(uchar3 * inPixels, int width, int height, int desiredWidth, uchar3 * outPixels, dim3 blockSize) {
    GpuTimer timer;
    timer.Start();
    // allocate kernel memory
    uchar3 * d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_energy;
    CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));
    int * d_leastSignificantPixel;
    CHECK(cudaMalloc(&d_leastSignificantPixel, height * sizeof(int)));
    int * d_minimalEnergy;
    CHECK(cudaMalloc(&d_minimalEnergy, width * height * sizeof(int)));

    // allocate host memory
    int * energy = (int *)malloc(width * height * sizeof(int));
    int * leastSignificantPixel = (int *)malloc(height * sizeof(int));
    int * minimalEnergy = (int *)malloc(width * height * sizeof(int));

    // dynamically sized smem used to compute energy
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);
    
    // block size use to calculate minimal energy to the end
    int blockSizeDp = 256;
    int gridSizeDp = (((width - 1) / blockSizeDp + 1) << 1) + 1;
    int stripHeight = (blockSizeDp >> 1) + 1;

    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // turn input image to grayscale
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    while (width > desiredWidth) {
        // update energy
        calEnergy<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_energy);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // compute min seam table
        for (int i = 0; i < height; i += (stripHeight >> 1)) {
            energyToTheEndKernel<<<gridSizeDp, blockSizeDp>>>(d_energy, d_minimalEnergy, width, height, i);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
        }

        // find least significant pixel index of each row and store in d_leastSignificantPixel (SEQUENTIAL, in kernel or host)
        CHECK(cudaMemcpy(minimalEnergy, d_minimalEnergy, WIDTH * height * sizeof(int), cudaMemcpyDeviceToHost));
        findSeam(minimalEnergy, leastSignificantPixel, width, height);
        // findSeamKernel<<<gridSize,blockSize>>>(d_minimalEnergy, d_leastSignificantPixel, width, height);
        // cudaDeviceSynchronize();
        // CHECK(cudaGetLastError());

        // carve
        CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        carvingKernel3<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_energy, width);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, WIDTH * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_leastSignificantPixel));
    CHECK(cudaFree(d_minimalEnergy));

    free(minimalEnergy);
    free(leastSignificantPixel);
    free(energy);

    timer.Stop();
    timer.printTime((char *)"device");   
}

int main(int argc, char ** argv) {   

    int width, height, desiredWidth;
    uchar3 * rgbPic;
    dim3 blockSize(32, 32);

    // Check user's input
    checkInput(argc, argv, width, height, rgbPic, desiredWidth, blockSize);

    // HOST
    uchar3 * out_host = (uchar3 *)malloc(width * height * sizeof(uchar3));
    hostResizing(rgbPic, width, height, desiredWidth, out_host);

    // DEVICE
    uchar3 * out_device = (uchar3 *)malloc(width * height * sizeof(uchar3));
    deviceResizing(rgbPic, width, height, desiredWidth, out_device, blockSize);

    // Compute error
    printError((char * )"Error between device result and host result: ", out_host, out_device, width, height);

    // Write 2 results to files
    writePnm(out_host, desiredWidth, height, width, concatStr(argv[2], "_host.pnm"));
    writePnm(out_device, desiredWidth, height, width, concatStr(argv[2], "_device.pnm"));

    // Free memories
    free(rgbPic);
    free(out_host);
    free(out_device);
}