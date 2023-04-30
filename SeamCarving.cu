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


__device__ uint8_t calculateGrayValue(const uchar3& pixel) {
    return (uint8_t)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
}



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
__global__ void calEnergy2(uint8_t *inPixels, int width, int height, int *energy) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ uint8_t s_inPixels[];

    // Load data into shared memory
    int s_col = threadIdx.x - filterWidth / 2;
    int s_row = threadIdx.y - filterWidth / 2;

    for (int i = s_row; i < blockDim.y + filterWidth - 1; i += blockDim.y) {
        for (int j = s_col; j < blockDim.x + filterWidth - 1; j += blockDim.x) {
            int tmpRow = min(max(row + i, 0), height - 1);
            int tmpCol = min(max(col + j, 0), width - 1);
            s_inPixels[(i + filterWidth / 2) * (blockDim.x + filterWidth - 1) + j + filterWidth / 2] = inPixels[tmpRow * width + tmpCol];
        }
    }

    __syncthreads();

    if (col < width && row < height) {
        int x_kernel = 0, y_kernel = 0;
        for (int i = 0; i < filterWidth; ++i) {
            for (int j = 0; j < filterWidth; ++j) {
                uint8_t closest = s_inPixels[(threadIdx.y + i) * (blockDim.x + filterWidth - 1) + threadIdx.x + j];
                int filterIdx = i * filterWidth + j;
                x_kernel += closest * d_xSobel[filterIdx];
                y_kernel += closest * d_ySobel[filterIdx];
            }
        }
        energy[row * width + col] = abs(x_kernel) + abs(y_kernel);
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
        readRow = min(max(readRow, 0), height - 1);//0 <= readCol <= height-1
        
        readCol = firstReadCol;
        virtualCol = threadIdx.x;

        for (; virtualCol < s_width; readCol += blockDim.x, virtualCol += blockDim.x) {
            tmpCol = readCol;
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


__global__ void findSeamAndRemoveKernel(uchar3 *outPixels, uint8_t *grayPixels, int * energy, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int r = idx / width;
    int c = idx % width;

    // Tính minimal energy trên mỗi hàng của ảnh bằng kỹ thuật dynamic programming
    if (r > 0) {
        int minPrevEnergy = energy[(r-1)*width+c];
        if (c > 0) {
            minPrevEnergy = min(minPrevEnergy, energy[(r-1)*width+c-1]);
        }
        if (c < width - 1) {
            minPrevEnergy = min(minPrevEnergy, energy[(r-1)*width+c+1]);
        }
        energy[idx] = grayPixels[idx] + minPrevEnergy;
    }


    // Tìm kiếm seam và loại bỏ chúng bằng cách sử dụng minimal energy đã tính được
    if (r < height - 1) {
        __syncthreads();
        int minCol = c;
        if (c > 0 && energy[(r+1)*width+c-1] < energy[r*width+minCol]) {
            minCol = c - 1;
        }
        if (c < width - 1 && energy[(r+1)*width+c+1] < energy[r*width+minCol]) {
            minCol = c + 1;
        }
        __syncthreads();
        if (minCol == c) {
            outPixels[idx] = outPixels[idx + width];
            grayPixels[idx] = grayPixels[idx + width];
            energy[idx] = energy[idx + width];
        }
    }

}

__global__ void carvingKernel1(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    int row = blockIdx.x;
    int baseIdx = row * d_WIDTH;
    for (int i = leastSignificantPixel[row]; i < width - 1; ++i) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        energy[baseIdx + i] = energy[baseIdx + i + 1];
    }
}

__global__ void carvingKernel2(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    int row = blockIdx.x;
    int leastSignificant = leastSignificantPixel[row];

    // Update pixels only for threads whose index is within the range of leastSignificant to width - 1
    for (int i = leastSignificant + threadIdx.x; i < width - 1; i += blockDim.x) {
        int baseIdx = row * d_WIDTH + i;
        outPixels[baseIdx] = outPixels[baseIdx + 1];
        grayPixels[baseIdx] = grayPixels[baseIdx + 1];
        energy[baseIdx] = energy[baseIdx + 1];
    }
}

__global__ void carvingKernel(int * leastSignificantPixel, uchar3 * outPixels, uint8_t *grayPixels, int * energy, int width) {
    __shared__ uchar3 sharedOutPixels[32];
    __shared__ uint8_t sharedGrayPixels[32];
    __shared__ int sharedEnergy[32];

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

__global__ void carvingKernel4(int *leastSignificantPixel, uchar3 *outPixels, uint8_t *grayPixels, int *energy, int width) {
    __shared__ uchar3 sharedOutPixels[32];
    __shared__ uint8_t sharedGrayPixels[32];
    __shared__ int sharedEnergy[32];

    int row = blockIdx.x;
    int leastSignificant = leastSignificantPixel[row];
    int baseIdx = row * d_WIDTH + leastSignificant;
    int idx = baseIdx + threadIdx.x;

    // Load data into shared memory
    sharedOutPixels[threadIdx.x] = outPixels[idx];
    sharedGrayPixels[threadIdx.x] = grayPixels[idx];
    sharedEnergy[threadIdx.x] = energy[idx];

    __syncthreads();

    // Update pixels in the current row
    for (int i = leastSignificant + threadIdx.x; i < width - 1; i += blockDim.x) {
        outPixels[baseIdx + i] = outPixels[baseIdx + i + 1];
        grayPixels[baseIdx + i] = grayPixels[baseIdx + i + 1];
        energy[baseIdx + i] = energy[baseIdx + i + 1];
    }

    __syncthreads();

    // Write updated data back to global memory
    outPixels[idx] = sharedOutPixels[threadIdx.x];
    grayPixels[idx] = sharedGrayPixels[threadIdx.x];
    energy[idx] = sharedEnergy[threadIdx.x];
}


__global__ void findSeamKernel2(int *minimalEnergy, int *leastSignificantPixel, int width, int height) {
    int col = threadIdx.x;
    int aboveIdx, leftIdx, rightIdx;
    int currEnergy, leftEnergy, rightEnergy;

    for (int row = height - 1; row >= 0; --row) {
        aboveIdx = (row - 1) * d_WIDTH + col;
        currEnergy = minimalEnergy[row * d_WIDTH + col];
        leftEnergy = col > 0 ? minimalEnergy[aboveIdx + (col - 1)] : INT_MAX;
        rightEnergy = col < width - 1 ? minimalEnergy[aboveIdx + (col + 1)] : INT_MAX;

        if (leftEnergy <= currEnergy && leftEnergy <= rightEnergy) {
            col -= 1;
        } else if (rightEnergy <= currEnergy && rightEnergy <= leftEnergy) {
            col += 1;
        }
        leastSignificantPixel[row] = col;
    }
}

__global__ void findSeamKernel(int * minimalEnergy, int *leastSignificantPixel, int width, int height) {
    int minCol = 0, r = height - 1;

    for (int c = 1; c < width; ++c)
        if (minimalEnergy[r * d_WIDTH + c] < minimalEnergy[r * d_WIDTH + minCol])
            minCol = c;
    
    for (; r >= 0; --r) {
        leastSignificantPixel[r] = minCol;
        if (r > 0) {
            int aboveIdx = (r - 1) * d_WIDTH + minCol;
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
        // CHECK(cudaMemcpy(minimalEnergy, d_minimalEnergy, WIDTH * height * sizeof(int), cudaMemcpyDeviceToHost));
        // findSeam(minimalEnergy, leastSignificantPixel, width, height);
        int numThreadsPerBlock = 256;
        int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        findSeamKernel<<<numBlocks, numThreadsPerBlock>>>(d_minimalEnergy, d_leastSignificantPixel, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());

        // carve    
        // CHECK(cudaMemcpy(d_leastSignificantPixel, leastSignificantPixel, height * sizeof(int), cudaMemcpyHostToDevice));
        carvingKernel<<<height, 1>>>(d_leastSignificantPixel, d_inPixels, d_grayPixels, d_energy, width);
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

void deviceResizing2(uchar3 * inPixels, int width, int height, int desiredWidth, uchar3 * outPixels, dim3 blockSize) {
    GpuTimer timer;
    timer.Start();
    // allocate kernel memory
    uchar3 * d_inPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    uint8_t * d_grayPixels;
    CHECK(cudaMalloc(&d_grayPixels, width * height * sizeof(uint8_t)));
    int * d_energy;
    CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));
    // allocate host memory
    int * energy = (int *)malloc(width * height * sizeof(int));
    // copy input to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));

    // turn input image to grayscale
    dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
    size_t smemSize = ((blockSize.x + 3 - 1) * (blockSize.y + 3 - 1)) * sizeof(uint8_t);
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    int numSeamsToRemove = abs(width - desiredWidth);
    calEnergy<<<gridSize, blockSize, smemSize>>>(d_grayPixels, width, height, d_energy);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    for (int i = 0; i < numSeamsToRemove; ++i) {
        findSeamAndRemoveKernel<<<gridSize,blockSize>>>(d_inPixels, d_grayPixels, d_energy, width, height);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        --width;
    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, WIDTH * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_grayPixels));
    CHECK(cudaFree(d_energy));

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