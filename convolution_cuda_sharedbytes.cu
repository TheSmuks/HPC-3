#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MAX_KERNEL_ELEMENTS 12544

__constant__ float d_const_kernel[MAX_KERNEL_ELEMENTS];

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

struct imagenppm{
    int altura;
    int ancho;
    char *comentario;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* ImagenData;

struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

ImagenData initimage(char* nombre, FILE **fp);
ImagenData duplicateImageData(ImagenData src);

int readImage(ImagenData Img, FILE *fp);
int readImageFast(int *R, int *G, int *B, FILE *fp, int pixel_count, int maxcolor);
int initfilestore(ImagenData img, FILE **fp, char* nombre);
int saveImage(ImagenData img, FILE **fp, int dim);
void freeImagestructure(ImagenData *src);

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

ImagenData initimage(char* nombre, FILE **fp){
    char c;
    char comentario[300];
    int i=0;
    ImagenData img=NULL;

    if ((*fp=fopen(nombre,"r"))==NULL){
        perror("Error: ");
    }
    else{

        img=(ImagenData) malloc(sizeof(struct imagenppm));

        fscanf(*fp,"%c%d ",&c,&(img->P));

        while((c=fgetc(*fp))!= '\n'){comentario[i]=c;i++;}
        comentario[i]='\0';

        img->comentario = (char *)calloc(strlen(comentario), sizeof(char));
        strcpy(img->comentario,comentario);

        fscanf(*fp,"%d %d %d",&img->ancho,&img->altura,&img->maxcolor);

        int total = img->ancho * img->altura;
        if ((img->R = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
        if ((img->G = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
        if ((img->B = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
    }
    return img;
}

ImagenData duplicateImageData(ImagenData src){

    ImagenData dst=(ImagenData) malloc(sizeof(struct imagenppm));

    dst->P=src->P;

    dst->comentario = (char *)calloc(strlen(src->comentario), sizeof(char));
    strcpy(dst->comentario,src->comentario);

    dst->ancho=src->ancho;
    dst->altura=src->altura;
    dst->maxcolor=src->maxcolor;

    int total = dst->ancho * dst->altura;
    if ((dst->R = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
    if ((dst->G = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
    if ((dst->B = (int *)calloc(total, sizeof(int))) == NULL) {return NULL;}
    return dst;
}

int readImageFast(int *R, int *G, int *B, FILE *fp, int pixel_count, int maxcolor){

    int digits = 1;
    int mc = (maxcolor > 0) ? maxcolor : 255;
    while (mc >= 10) { mc /= 10; digits++; }
    long bytes_per_value = (long)digits + 2;

    long buf_size = (long)pixel_count * 3 * bytes_per_value + 1024;
    char *buf = (char *)malloc(buf_size + 1);
    if (!buf) {
        return -1;
    }

    long bytes_read = fread(buf, 1, buf_size, fp);
    buf[bytes_read] = '\0';

    char *ptr = buf;
    for (int i = 0; i < pixel_count; i++) {
        R[i] = (int)strtol(ptr, &ptr, 10);
        G[i] = (int)strtol(ptr, &ptr, 10);
        B[i] = (int)strtol(ptr, &ptr, 10);
    }

    free(buf);
    return 0;
}

int readImage(ImagenData img, FILE *fp){
    return readImageFast(img->R, img->G, img->B, fp,
                         img->ancho * img->altura, img->maxcolor);
}

kernelData leerKernel(char* nombre){
    FILE *fp;
    int i=0;
    kernelData kern=NULL;

    fp=fopen(nombre,"r");
    if(!fp){
        perror("Error: ");
    }
    else{

        kern=(kernelData) malloc(sizeof(struct structkernel));

        fscanf(fp,"%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX*kern->kernelY*sizeof(float));

        for (i=0;i<(kern->kernelX*kern->kernelY)-1;i++){
            fscanf(fp,"%f,",&kern->vkern[i]);
        }
        fscanf(fp,"%f",&kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

int initfilestore(ImagenData img, FILE **fp, char* nombre){

    if ( (*fp=fopen(nombre,"w")) == NULL ){
        perror("Error: ");
        return -1;
    }

    fprintf(*fp,"P%d\n%s\n%d %d\n%d\n",img->P,img->comentario,img->ancho,img->altura,img->maxcolor);
    return 0;
}

int saveImage(ImagenData img, FILE **fp, int dim){
    int i;
    for(i=0;i<dim;i++){
        fprintf(*fp,"%d %d %d ",img->R[i],img->G[i],img->B[i]);
    }
    return 0;
}

void freeImagestructure(ImagenData *src){

    free((*src)->comentario);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);

    free(*src);
}

__global__ void convolve2D_shared(const int* input, int* output,
                                  int dataSizeX, int dataSizeY,
                                  int kernelSizeX, int kernelSizeY)
{
    extern __shared__ int sharedTile[];

    int kernelCenterX = kernelSizeX / 2;
    int kernelCenterY = kernelSizeY / 2;
    int tileWidth  = blockDim.x + 2 * kernelCenterX;
    int tileHeight = blockDim.y + 2 * kernelCenterY;
    int blockOriginX = blockIdx.x * blockDim.x;
    int blockOriginY = blockIdx.y * blockDim.y;

    for (int tileRow = threadIdx.y; tileRow < tileHeight; tileRow += blockDim.y) {
        int globalY = blockOriginY - kernelCenterY + tileRow;
        for (int tileCol = threadIdx.x; tileCol < tileWidth; tileCol += blockDim.x) {
            int globalX = blockOriginX - kernelCenterX + tileCol;
            int value = 0;
            if (globalX >= 0 && globalX < dataSizeX && globalY >= 0 && globalY < dataSizeY)
                value = input[globalY * dataSizeX + globalX];
            sharedTile[tileRow * tileWidth + tileCol] = value;
        }
    }
    __syncthreads();

    int outputX = blockOriginX + threadIdx.x;
    int outputY = blockOriginY + threadIdx.y;
    if (outputX >= dataSizeX || outputY >= dataSizeY) return;

    int sum = 0;
    for (int kernelRow = 0; kernelRow < kernelSizeY; ++kernelRow) {
        int sharedRow = threadIdx.y + 2 * kernelCenterY - kernelRow;
        for (int kernelCol = 0; kernelCol < kernelSizeX; ++kernelCol) {
            int sharedCol = threadIdx.x + 2 * kernelCenterX - kernelCol;
            sum += sharedTile[sharedRow * tileWidth + sharedCol] *
                   d_const_kernel[kernelRow * kernelSizeX + kernelCol];
        }
    }

    output[outputY * dataSizeX + outputX] = (sum >= 0) ? (int)(sum + 0.5f)
                                                       : (int)(sum - 0.5f);
}

__global__ void convolve2D_global(const int* input, int* output,
                                  int dataSizeX, int dataSizeY,
                                  int kernelSizeX, int kernelSizeY)
{
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX >= dataSizeX || outputY >= dataSizeY) return;

    int kernelCenterX = kernelSizeX / 2;
    int kernelCenterY = kernelSizeY / 2;

    int sum = 0;
    for (int kernelRow = 0; kernelRow < kernelSizeY; ++kernelRow) {
        int inputRow = outputY + (kernelCenterY - kernelRow);
        if (inputRow < 0 || inputRow >= dataSizeY) continue;

        for (int kernelCol = 0; kernelCol < kernelSizeX; ++kernelCol) {
            int inputCol = outputX + (kernelCenterX - kernelCol);
            if (inputCol < 0 || inputCol >= dataSizeX) continue;

            sum += input[inputRow * dataSizeX + inputCol] *
                   d_const_kernel[kernelRow * kernelSizeX + kernelCol];
        }
    }

    output[outputY * dataSizeX + outputX] = (sum >= 0) ? (int)(sum + 0.5f)
                                                       : (int)(sum - 0.5f);
}

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 8)
    {
        printf("Usage: %s <image-file> <kernel-file> <result-file> [blockX blockY gridX gridY]\n", argv[0]);
        printf("\n\nError, Missing parameters:\n");
        printf("format: ./convolution_cuda image_file kernel_file result_file [blockX blockY gridX gridY]\n");
        printf("- image_file : source image path (*.ppm)\n");
        printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
        printf("- result_file: result image path (*.ppm)\n\n");
        printf("- blockX/blockY : CUDA threads per block dimensions. Default: 16 16\n");
        printf("- gridX/gridY   : CUDA grid dimensions. Use values that cover the image. Default: ceil(image/block)\n\n");
        return -1;
    }

    int imagesize;
    double start, tstart=0, tend=0, tread=0, tcopy=0, tconv=0, tkernel=0, tstore=0, treadk=0;
    FILE *fpsrc=NULL,*fpdst=NULL;
    ImagenData source=NULL, output=NULL;

    int blockX = 16, blockY = 16;
    int userGridX = 0, userGridY = 0;
    if (argc == 8) {
        blockX = atoi(argv[4]);
        blockY = atoi(argv[5]);
        userGridX = atoi(argv[6]);
        userGridY = atoi(argv[7]);
        if (blockX <= 0 || blockY <= 0 || userGridX <= 0 || userGridY <= 0) {
            fprintf(stderr, "Error: blockX, blockY, gridX and gridY must all be positive integers.\n");
            return -1;
        }
    }
    if (blockX * blockY > 1024) {
        fprintf(stderr, "Error: blockX*blockY = %d exceeds the 1024 threads-per-block limit.\n",
                blockX * blockY);
        return -1;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int maxSharedPerBlock = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedPerBlock,
                                      cudaDevAttrMaxSharedMemoryPerBlock, dev));

    start = wall_time();
    tstart = start;
    kernelData kern=NULL;
    if ( (kern = leerKernel(argv[2]))==NULL) {
        return -1;
    }
    treadk = treadk + (wall_time() - start);

    int kernelCenterX = kern->kernelX / 2;
    int kernelCenterY = kern->kernelY / 2;
    int tileWidth  = blockX + 2 * kernelCenterX;
    int tileHeight = blockY + 2 * kernelCenterY;
    size_t sharedBytes = (size_t)tileWidth * (size_t)tileHeight * sizeof(int);
    int useTiled = (sharedBytes <= (size_t)maxSharedPerBlock);

    start = wall_time();
    if ( (source = initimage(argv[1], &fpsrc)) == NULL) {
        return -1;
    }
    tread = tread + (wall_time() - start);

    start = wall_time();
    if ( (output = duplicateImageData(source)) == NULL) {
        return -1;
    }
    tcopy = tcopy + (wall_time() - start);

    start = wall_time();
    if (initfilestore(output, &fpdst, argv[3])!=0) {
        perror("Error: ");
        return -1;
    }
    tstore = tstore + (wall_time() - start);

    imagesize = source->altura * source->ancho;

    size_t kernel_bytes = (size_t)kern->kernelX * (size_t)kern->kernelY * sizeof(float);
    if ((size_t)kern->kernelX * (size_t)kern->kernelY > MAX_KERNEL_ELEMENTS) {
        fprintf(stderr, "Error: kernel has %zu elements, but constant memory buffer supports at most %d.\n",
                (size_t)kern->kernelX * (size_t)kern->kernelY, MAX_KERNEL_ELEMENTS);
        return -1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, kern->vkern, kernel_bytes));

    start = wall_time();
    if (readImage(source, fpsrc)) {
        return -1;
    }
    tread = tread + (wall_time() - start);

    start = wall_time();
    int dataSizeX = source->ancho;
    int dataSizeY = source->altura;
    size_t channel_bytes = (size_t)imagesize * sizeof(int);
    int *d_inputR = NULL, *d_outputR = NULL;
    int *d_inputG = NULL, *d_outputG = NULL;
    int *d_inputB = NULL, *d_outputB = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_inputR, channel_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_outputR, channel_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inputG, channel_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_outputG, channel_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_inputB, channel_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_outputB, channel_bytes));

    CUDA_CHECK(cudaMemcpy(d_inputR, source->R, channel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputG, source->G, channel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inputB, source->B, channel_bytes, cudaMemcpyHostToDevice));

    dim3 block(blockX, blockY);
    dim3 grid((argc == 8) ? userGridX : ((dataSizeX + block.x - 1) / block.x),
              (argc == 8) ? userGridY : ((dataSizeY + block.y - 1) / block.y));
    if (grid.x * block.x < (unsigned int)dataSizeX || grid.y * block.y < (unsigned int)dataSizeY) {
        fprintf(stderr,
                "Error: launch geometry grid=(%u,%u), block=(%u,%u) covers only %u x %u pixels, but image needs %d x %d.\n",
                grid.x, grid.y, block.x, block.y,
                grid.x * block.x, grid.y * block.y, dataSizeX, dataSizeY);
        return -1;
    }

    cudaEvent_t eventStart, eventStop;
    CUDA_CHECK(cudaEventCreate(&eventStart));
    CUDA_CHECK(cudaEventCreate(&eventStop));

    CUDA_CHECK(cudaEventRecord(eventStart));
    if (useTiled) {
        convolve2D_shared<<<grid, block, sharedBytes>>>(d_inputR, d_outputR, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
        convolve2D_shared<<<grid, block, sharedBytes>>>(d_inputG, d_outputG, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
        convolve2D_shared<<<grid, block, sharedBytes>>>(d_inputB, d_outputB, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
    } else {
        convolve2D_global<<<grid, block>>>(d_inputR, d_outputR, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
        convolve2D_global<<<grid, block>>>(d_inputG, d_outputG, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
        convolve2D_global<<<grid, block>>>(d_inputB, d_outputB, dataSizeX, dataSizeY, kern->kernelX, kern->kernelY);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(eventStop));
    CUDA_CHECK(cudaEventSynchronize(eventStop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, eventStart, eventStop));
    tkernel = tkernel + ((double)kernel_ms / 1000.0);
    CUDA_CHECK(cudaEventDestroy(eventStart));
    CUDA_CHECK(cudaEventDestroy(eventStop));

    CUDA_CHECK(cudaMemcpy(output->R, d_outputR, channel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output->G, d_outputG, channel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output->B, d_outputB, channel_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_inputR));
    CUDA_CHECK(cudaFree(d_outputR));
    CUDA_CHECK(cudaFree(d_inputG));
    CUDA_CHECK(cudaFree(d_outputG));
    CUDA_CHECK(cudaFree(d_inputB));
    CUDA_CHECK(cudaFree(d_outputB));
    tconv = tconv + (wall_time() - start);

    start = wall_time();
    if (saveImage(output, &fpdst, imagesize)) {
        perror("Error: ");
        return -1;
    }
    tstore = tstore + (wall_time() - start);

    fclose(fpsrc);
    fclose(fpdst);

    tend = wall_time();

    printf("Imatge: %s\n", argv[1]);
    printf("ISizeX : %d\n", source->ancho);
    printf("ISizeY : %d\n", source->altura);
    printf("kSizeX : %d\n", kern->kernelX);
    printf("kSizeY : %d\n", kern->kernelY);
    printf("blockX : %d\n", blockX);
    printf("blockY : %d\n", blockY);
    printf("gridX  : %d\n", (argc == 8) ? userGridX : ((source->ancho + blockX - 1) / blockX));
    printf("gridY  : %d\n", (argc == 8) ? userGridY : ((source->altura + blockY - 1) / blockY));
    printf("CUDA kernel impl: %s\n", useTiled ? "shared" : "global");
    printf("CUDA shared tile bytes: %zu (device max per block: %d)\n", sharedBytes, maxSharedPerBlock);
    printf("%.6lf seconds elapsed for Reading image file.\n", tread);
    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
    printf("%.6lf seconds elapsed for Reading kernel matrix.\n", treadk);
    printf("%.6lf seconds elapsed for CUDA allocation, transfers, kernel launches and cleanup.\n", tconv);
    printf("%.6lf seconds elapsed for CUDA kernel execution only.\n", tkernel);
    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
    printf("%.6lf seconds elapsed\n", tend-tstart);

    freeImagestructure(&source);
    freeImagestructure(&output);

    return 0;
}
