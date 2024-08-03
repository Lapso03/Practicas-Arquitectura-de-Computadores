/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 * Curso:
 *
 * ENTREGA no.#3# <Procesamiento de imágenes>
 *
 * EQUIPO: TE-C 24
 * MIEMBROS: Luna Tejedor, Sergio Buil, Carlos Gimeno, Ibai Moya
 *
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "gpu_bitmap.h"
__global__ void kernel(unsigned char *imagen)
{
    float R, G, B, Y;
    // ** Kernel bidimensional multibloque **
    // coordenada horizontal de cada hilo
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    // coordenada vertical de cada hilo
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // indice global de cada hilo (indice lineal para acceder a la memoria)
    int myID = x + y * blockDim.x * gridDim.x;

    // cada hilo obtiene la posicion de su pixel
    int miPixel = myID * 4;

    // cada hilo rellena los 4 canales de su pixel con un valor arbitrario
    R = imagen[miPixel + 0]; // canal R
    G = imagen[miPixel + 1]; // canal G
    B = imagen[miPixel + 2]; // canal B
    Y = 0.299 * R + 0.587 * G + 0.114 * B;

    imagen[miPixel + 0] = Y; // canal R
    imagen[miPixel + 1] = Y; // canal G
    imagen[miPixel + 2] = Y; // canal B
    imagen[miPixel + 3] = 0; // canal alfa
}

__host__ void leerBMP_RGBA(const char *nombre, int *w, int *h, unsigned char **imagen);

__host__ void propiedades_Device(int deviceID);
{
    int runtimeVersion;

    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);

    // Cálculo del número de cores
    int cudaCores = 0;
    int SM = deviceProp.multiProcessorCount;
    int major = deviceProp.major;
    int minor = deviceProp.minor;
    const char *archName;

    switch (major)
    {
    case 1: // Tesla
        archName = "Tesla";
        cudaCores = 8;
        break;
    case 2: // Fermi
        archName = "Fermi";
        if (minor == 1)
        {
            cudaCores = 48;
        }
        else
        {
            cudaCores = 32;
        }
        break;
    case 3: // Kepler
        archName = "Kepler";
        cudaCores = 192;
        break;
    case 5: // Maxwell
        archName = "Maxwell";
        cudaCores = 128;
        break;
    case 6: // Pascal
        archName = "Pascal";
        cudaCores = 64;
        break;

    case 7: // Volta (7.0) Turing (7.5)
        cudaCores = 64;
        if (minor == 0)
        {
            archName = "Volta";
        }
        else
        {
            archName = "Turing";
        }
        break;
    case 8: // Ampere
        archName = "Ampere";
        cudaCores = 64;
        break;

    default: // Arquitectura desconocida
        archName = "Desconocida";
        cudaCores = 0;
        break;
    }

    // Presentación de propiedades
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
    printf("***************************************************\n");
    printf("> CUDA Toolkit \t: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("> Arquitectura CUDA \t: %s\n", archName);
    printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
    printf("> No. de MultiProcesadores \t: %d\n", SM);
    printf("> No. de CUDA Cores (%dx%d) \t: %d\n", cudaCores, SM, cudaCores * SM);
    printf("> No. maximo de hilos por bloque: %d\n", deviceProp.maxThreadsPerBlock);
    printf("> Memoria global: %d\n", deviceProp.totalGlobalMem);
    printf("***************************************************\n");
}

// Main
int main(int argc, char **argv)
{
    int currentDevice;
    cudaGetDevice(&currentDevice);
    propiedades_Device(currentDevice);

    // Declaración de eventos
    cuda_Event_t start, stop;

    // Creación de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Leemos el archivo BMP
    unsigned char *host_color;
    int ancho, alto;
    leerBMP_RGBA("imagen.bmp", &ancho, &alto, &host_color);

    // Declaración del bitmap RGBA:

    // Inicialización de la estructura RenderGPU
    RenderGPU foto(ancho, alto);

    // Tamaño del bitmap en bytes
    size_t img_size = ancho * alto * 4 * sizeof(unsigned char);

    // Asignación y reserva de la memoria en el host(framebuffer)
    unsigned char *host_bitmap = foto.get_ptr();

    unsigned char *dev_color;

    cudaMalloc((void **)&dev_color, img_size);

    // Lanzamos un kernel bidimensional con bloques de 256 hilos (20x20)
    dim3 hilosB(20, 20);

    // Calculamos el número de bloques necesario (un hilo por cada pixel)
    dim3 Nbloques(ancho / 20, alto / 20);

    // Marca de inicio
    cudaEventRecord(start, 0);

    // Generamos el bitmap
    cudaMemcpy(dev_color, host_color, img_size, cudaMemcpyHostToDevice);
    kernel<<<Nbloques, hilosB>>>(dev_color);
    cudaMemcpy(host_bitmap, dev_color, img_size, cudaMemcpyDeviceToHost);

    // Marca de final
    cudaEventRecord(stop, 0);

    // Sincronización GPU-CPU
    cudaEventSynchronize(stop);

    // Cálculo del tiempo en ms
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Liberación de recursos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Información del sistema
    char infoName[1024];
    char infoUser[1024];
    DWORD longitud;
    GetComputerName(infoName, &longitud);
    GetUserName(infoUser, &longitud);

    time_t fecha;
    time(&fecha);

    // Impresión de resultados (tiempo GPU)
    printf("***************************************************\n");
    printf("> Tiempo de ejecución: %f ms\n", elapsed_time);
    printf(">KERNEL DE %d BLOQUE(S) con %d HILOS: \n", ancho / 20 * alto / 20, ancho * alto);
    printf("Eje X -> %d bloques\n", ancho / 20);
    printf("Eje Y -> %d bloques\n", alto / 20);
    printf("Programa ejecutado el día: %s", ctime(&fecha));
    printf("Máquina: %s\n", infoName);
    printf("Usuario: %s\n", infoUser);
    printf("***************************************************\n");

    printf("\n...pulsa [ESC] para finalizar...");
    foto.display_and_exit();
    return 0;
}

__host__ void leerBMP_RGBA(const char *nombre, int *w, int *h, unsigned char **imagen)
{
    FILE *archivo;

    // Abrimos el archivo en modo lectura binaria
    if ((archivo = fopen(nombre, "rb")) == NULL)
    {
        printf("Error al abrir el archivo %s\n", nombre);
        printf("...pulsa [INTRO] para finalizar...");
        getchar();
        exit(1);
    }
    printf("Leyendo archivo %s\n", nombre);

    // Leemos la cabecera del archivo BMP (54 bytes -> 14 cabecera + 40 info)
    unsigned char tipo[2];

    fread(tipo, 1, 2, archivo);

    // Comprobamos si es un archivo BMP
    if (tipo[0] != 'B' || tipo[1] != 'M')
    {
        printf("El archivo %s no es un archivo BMP\n", nombre);
        printf("...pulsa [INTRO] para finalizar...");
        getchar();
        exit(1);
    }

    // Leemos el tamaño del archivo (leemos 4 bytes)
    unsigned int file_size;
    fread(&file_size, 4, 1, archivo);

    // Leemos 4 bytes reservados
    unsigned char buffer[4];
    fread(&reservado, 1, 4, archivo);

    // Leemos el offset de la imagen
    unsigned int offset;
    fread(&offset, 4, 1, archivo);

    // Imprimimos la información
    printf("***************************************************\n");
    printf("Datos de la cabecera BMP\n");
    printf("> Tipo de archivo: %c%c\n", tipo[0], tipo[1]);
    printf("> Tamaño del archivo: %u KiB\n", file_size);
    printf("> Offset de datos: %u bytes\n", offset);
    printf("***************************************************\n");

    // Leemos la cabecera de información de la imagen (40 bytes)
    unsigned int header_size;
    fread(&header_size, 4, 1, archivo);

    // Leemos el anchos y alto de la imagen
    unsigned int ancho;
    unsigned int alto;
    fread(&ancho, 4, 1, archivo);
    fread(&alto, 4, 1, archivo);

    // Leemos el número de planos
    unsigned short int planos;
    fread(&planos, 2, 1, archivo);

    // Leemos la profundidad del color
    unsigned short int color_depth;
    fread(&color_depth, 2, 1, archivo);

    // Leemos el tipo de compresión
    unsigned int compresion;
    fread(&compresion, 4, 1, archivo);

    // Imprimimos la información
    printf("***************************************************\n");
    printf("Datos de la cabecera de información\n");
    printf("> Tamaño de la cabecera: %u bytes\n", header_size);
    printf("> Dimensiones de la imagen: %u x %u\n", ancho, alto);
    printf("> Planos: %d\n", planos);
    printf("> Profundidad del color: %d bits\n", color_depth);
    printf("> Tipo de compresión: %s\n", (compresion == 0) ? "none" : "Con compresión");
    printf("***************************************************\n");

    // Leemos los datos del archivo
    size_t img_size = ancho * alto * 4;

    unsigned char *datos = (unsigned char *)malloc(img_size);

    // Nos situamos en el offset de la imagen
    fseek(archivo, offset, SEEK_SET);

    // Leemos los datos de la imagen pixel a pixel
    unsigned int pixel_size = color_depth / 8;

    for (unsigned int i = 0; i < ancho * alto; i++)
    {
        fread(buffer, 1, pixel_size, archivo); // Leemos los datos de un pixel
        datos[i * 4] = buffer[2];              // R
        datos[i * 4 + 1] = buffer[1];          // G
        datos[i * 4 + 2] = buffer[0];          // B
        datos[i * 4 + 3] = 255;                // A
    }

    // Cerramos el archivo
    fclose(archivo);

    // Asignamos los valores a los punteros
    *w = ancho;
    *h = alto;
    *imagen = datos;

    printf("Archivo %s leído correctamente\n", nombre);

    return;
}