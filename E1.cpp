/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 * Curso: 2º
 *
 * ENTREGA no.#1# <Temporización GPU>
 *
 * EQUIPO: TE-C 24
 * MIEMBROS: Luna Tejedor, Sergio Buil, Carlos Gimeno, Ibai Moya
 *
 *
 *
 */
///////////////////////////////////////////////////////////////////////////
// includes
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> //carga el tiempo
// Funcion que imprime las propiedades del device
__host__ void propiedades_Device(int deviceID);
// cambia las filas de la matriz
__global__ void matriz(int *dev_A, int *dev_B, int C, int F);
// genera una matriz de numeros aleatorios entre 1 y 9
__host__ void matrizNumerosAleatorios(int *hst_A, int C, int F);
////////////////////////////////////////////////////////////////////
// MAIN
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

    // Declaraciones
    int F;
    printf("Introduce el numero de filas de la matriz: \n");
    scanf("%d", &F);
    getchar();
    int C;
    printf("Introduce el numero de columnas de la matriz: \n");
    scanf("%d", &C);
    getchar();

    // Reserva de memoria
    int *hst_A, *hst_B;
    int *dev_A, *dev_B;
    // En el HOST
    hst_A = (int *)malloc(F * C * sizeof(int));
    hst_B = (int *)malloc(F * C * sizeof(int));
    // En el DEVICE
    cudaMalloc((void **)&dev_A, F * C * sizeof(int));
    cudaMalloc((void **)&dev_B, F * C * sizeof(int));

    // Ejecucion en HOST
    matrizNumerosAleatorios(hst_A, F, C);
    // paso de detos del HOST al DEVICE
    cudaMemcpy(dev_A, hst_A, F * C * sizeof(int), cudaMemcpyHostToDevice);
    // Dimensiones del kernel
    dim3 Nbloques(1);
    dim3 hilosB(C, F); // En el eje X las columnas y en el eje Y las filas

    // Informacion del kernel
    printf("KERNEL DE 1 BLOQUE(S) con %d HILOS: \n", F * C);
    printf("[eje X -> %d]\n[eje Y -> %d]\n", C, F);

    // marca de inicio
    cudaEventRecord(start, 0);
    // EJECUCIÓN EN EL DEVICE
    matriz<<<Nbloques, hilosB>>>(dev_A, dev_B, C, F);
    // recogida de datos del DEVICE al HOST
    cudaMemcpy(hst_B, dev_B, F * C * sizeof(int), cudaMemcpyDeviceToHost);
    // marca de final
    cudaEventRecord(stop, 0);
    // sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    // calculo del tiempo en ms
    float tiempo_ms;
    cudaEventElapsedTime(&tiempo_ms, start, stop);

    // impresion de resultados (tiempo GPU)
    printf("> Tiempo de ejecucion: %f ms\n", tiempo_ms);
    // imprimir resultados
    printf("MATRIZ ORIGINAL:\n");
    for (int i = 0; i < F; i++)
    {
        for (int j = 0; j < C; j++)
        {
            printf("%2.1d ", hst_A[j + i * C]);
        }
        printf("\n");
    }
    printf("MATRIZ FINAL:\n");
    for (int i = 0; i < F; i++)
    {
        for (int j = 0; j < C; j++)
        {
            printf("%2.1d ", hst_B[j + i * C]);
        }
        printf("\n");
    }

    // Informacion del Sistema
    char infoName[1024];
    char infoUser[1024];
    DWORD longitud;
    GetComputerName(infoName, &longitud);
    GetUserName(infoUser, &longitud);
    time_t fecha;
    time(&fecha);
    printf("\n***************************************************\n");
    printf("Programa ejecutado el dia: %s", ctime(&fecha));
    printf("Maquina: %s\n", infoName);
    printf("Usuario: %s\n", infoUser);
    // SALIDA DEL PROGRAMA
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}

// CODIGO APORTADO DE LA PRACTICA
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
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
    printf("***************************************************\n");
    printf("> CUDA Toolkit \t: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("> Arquitectura CUDA \t: %s\n", archName);
    printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
    printf("> No. de MultiProcesadores \t: %d\n", SM);
    printf("> No. de CUDA Cores (%dx%d) \t: %d\n", cudaCores, SM, cudaCores * SM);
    printf("> No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
    printf("> Memoria Global (total) \t: %zu MiB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf("***************************************************\n");
}
// FIN DE CODIGO APORTADO DE LA PRACTICA

// Matriz inicial
__host__ void matrizNumerosAleatorios(int *hst_A, int F, int C)
{
    for (int i = 0; i < F; i++)
    {
        int numero = rand() % 9 + 1;
        for (int j = 0; j < C; j++)
        {
            hst_A[j + C * i] = numero;
        }
    }
}

// se copia la matriz inicial y se mueven las columnas
__global__ void matriz(int *dev_A, int *dev_B, int C, int F)
{
    int fila = threadIdx.y;
    int columna = threadIdx.x;
    // Indices lineales en el que movemos las filas
    int myID = columna + blockDim.x * fila;
    int myID2 = columna + blockDim.x * (fila + 1);
    int myID3 = columna + blockDim.x * (fila - (F - 1));
    // Cuando las filas sean menores que la última fila:
    if (fila < F - 1)
    {
        dev_B[myID2] = dev_A[myID];
    }
    // última fila:
    else
    {
        dev_B[myID3] = dev_A[myID];
    }
}