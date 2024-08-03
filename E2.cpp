/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 * Curso:
 *
 * ENTREGA no.#2# <Temporizacion GPU>
 *
 * EQUIPO: TE-C 24
 * MIEMBROS: Luna Tejedor, Sergio Buil, Carlos Gimeno, Ibai Moya
 *
 */

// includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"

// defines
#define ANCHO 480 // Dimension horizontal
#define ALTO 480  // Dimension vertical

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void kernel(unsigned char *imagen)
{
    // ** Kernel bidimensional multibloque **
    //
    // coordenada horizontal de cada hilo
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // coordenada vertical de cada hilo
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // indice global de cada hilo (indice lineal para acceder a la memoria)
    int myID = x + y * blockDim.x * gridDim.x;
    // cada hilo obtiene la posicion de su pixel
    int miPixel = myID * 4;
    // variables nuevas que aumentas el tamaño
    int maspixel_x = x / 60;
    int maspixel_y = y / 60;
    // cada hilo rellena los 4 canales de su pixel con un valor arbitrario
    if ((maspixel_x + maspixel_y) % 2 == 0) // Negro
    {
        imagen[miPixel + 0] = 0; // canal R
        imagen[miPixel + 1] = 0; // canal G
        imagen[miPixel + 2] = 0; // canal B
        imagen[miPixel + 3] = 0; // canal alfa
    }
    else // Blanco
    {
        imagen[miPixel + 0] = 255; // canal R
        imagen[miPixel + 1] = 255; // canal G
        imagen[miPixel + 2] = 255; // canal B
        imagen[miPixel + 3] = 255; // canal alfa
    }
}
__host__ void propiedades_Device(int deviceID);
// MAIN: rutina principal ejecutada en el host
int main(int argc, char **argv)
{
    // Dispositivo CUDA
    int currentDevice;
    cudaGetDevice(&currentDevice);
    propiedades_Device(currentDevice);
    // Declaraciones de eventos
    cudaEvent_t start;
    cudaEvent_t stop;
    // Creacion de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Declaracion del bitmap:
    // Inicializacion de la estructura RenderGPU
    RenderGPU foto(ANCHO, ALTO);
    // Tamaño del bitmap en bytes
    size_t size = foto.image_size();
    // Asignacion y reserva de la memoria en el host (framebuffer)
    unsigned char *host_bitmap = foto.get_ptr();
    // Reserva en el device
    unsigned char *dev_bitmap;
    cudaMalloc((void **)&dev_bitmap, size);
    // Lanzamos un kernel bidimensional con bloques de 256 hilos (16x16)
    dim3 hilosB(16, 16);
    // Calculamos el numero de bloques necesario (un hilo por cada pixel)
    dim3 Nbloques(ANCHO / 16, ALTO / 16);
    // marca de inicio
    cudaEventRecord(start, 0);
    // Generamos el bitmap
    kernel<<<Nbloques, hilosB>>>(dev_bitmap);
    // Copiamos los datos desde la GPU hasta el framebuffer para visualizarlos
    cudaMemcpy(host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost);
    // marca de final
    cudaEventRecord(stop, 0);
    // sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    // calculo del tiempo en ms
    float tiempo_ms;
    cudaEventElapsedTime(&tiempo_ms, start, stop);
    // impresion de resultados (tiempo GPU)
    printf("> Tiempo de ejecucion: %f ms\n", tiempo_ms);
    // Visualizacion y salida
    printf("\n...pulsa [ESC] para finalizar...");
    foto.display_and_exit();
    return 0;
}
__host__ void propiedades_Device(int deviceID)
{
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);
    // calculo del numero de cores (SP)
    int cudaCores = 0;
    int SM = deviceProp.multiProcessorCount;
    int major = deviceProp.major;
    int minor = deviceProp.minor;
    const char *archName;
    switch (major)
    {
    case 1:
        // TESLA
        archName = "TESLA";
        cudaCores = 8;
        break;
    case 2:
        // FERMI
        archName = "FERMI";
        if (minor == 0)
            cudaCores = 32;
        else
            cudaCores = 48;
        break;
    case 3:
        // KEPLER
        archName = "KEPLER";
        cudaCores = 192;
        break;
    case 5:
        // MAXWELL
        archName = "MAXWELL";
        cudaCores = 128;
        break;
    case 6:
        // PASCAL
        archName = "PASCAL";
        cudaCores = 64;
        break;
    case 7:
        // VOLTA (7.0) TURING (7.5)
        cudaCores = 64;
        if (minor == 0)
            archName = "VOLTA";
        else
            archName = "TURING";
        break;
    case 8:
        // AMPERE
        archName = "AMPERE";
        cudaCores = 64;
        break;
    default:
        // ARQUITECTURA DESCONOCIDA
        archName = "DESCONOCIDA";
        cudaCores = 0;
    }
    // presentacion de propiedades
    printf("*****************\n");
    printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
    printf("*****************\n");
    printf("> CUDA Toolkit \t: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("> Arquitectura CUDA \t: %s\n", archName);
    printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
    printf("> No. de MultiProcesadores \t: %d\n", SM);
    printf("> No. de CUDA Cores (%dx%d) \t: %d\n", cudaCores, SM,
           cudaCores * SM);
    printf("> No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
    printf("> Memoria Global (total) \t: %zu MiB\n",
           deviceProp.totalGlobalMem / (1024 * 1024));
    printf("*****************\n");
}