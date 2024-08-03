/*
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 * Curso: 2022-2023
 *
 * ENTREGA no.5 Rendimiento GPU vs CPU
 *
 * EQUIPO: tec-24
 * MIEMBROS: Ibai Moya, Luna Tejedor, Sergio Buil y Carlos Gimeno
 *
 */
///////////////////////////////////////////////////////////////////////////
// includes
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#ifdef __linux__
#include <sys/time.h>
typedef struct timeval event;
#else
#include <windows.h>
typedef LARGE_INTEGER event;
#endif
// HOST: funcion llamada desde el host y ejecutada en el host
__host__ void setEvent(event *ev)
/* Descripcion: Genera un evento de tiempo */
{
#ifdef __linux__
    gettimeofday(ev, NULL);
#else
    QueryPerformanceCounter(ev);
#endif
}
__host__ double eventDiff(event *first, event *last)
/* Descripcion: Devuelve la diferencia de tiempo (en ms) entre dos eventos */
{
#ifdef __linux__
    return ((double)(last->tv_sec + (double)last->tv_usec / 1000000) - (double)(first->tv_sec + (double)first->tv_usec / 1000000)) * 1000.0;
#else
    event freq;
    QueryPerformanceFrequency(&freq);
    return ((double)(last->QuadPart - first->QuadPart) / (double)freq.QuadPart) * 1000.0;
#endif
}
__global__ void ordenarVectorPorRango(int *vectorInicial, int *vectorOrdenado, int tam)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    if (myID < tam)
    {
        int rango = 0;
        for (int i = 0; i < tam; i++)
        {
            if (vectorInicial[myID] > vectorInicial[i])
            {
                rango++;
            }
            else
            {
                if (vectorInicial[myID] == vectorInicial[i] && myID > i)
                {
                    rango++;
                }
            }
        }
        vectorOrdenado[rango] = vectorInicial[myID];
    }
}
// Esta funcion que imprime las propiedades del device
__host__ int propiedades_Device(int deviceID);
////////////////////////////////////////////////////////////////////

// MAIN
int main(int argc, char **argv)
{
    // Dispositivo CUDA
    int currentDevice;
    cudaGetDevice(&currentDevice);
    int tammax = propiedades_Device(currentDevice);
    int tam = 0;
    while (!(tam > 0))
    {
        printf("Introduce el numero de elementos del vector a generar: ");
        scanf("%d", &tam);
        printf("\n");
    }
    int mostrar = 3;
    while (!(mostrar == 0 || mostrar == 1))
    {
        printf("Quieres ver los vectores por pantalla? \nSi --> 0 \nNo --> 1 \n");
        scanf("%d", &mostrar);
        printf("\n");
    }
    int *hst_vectorInicial, *hst_vectorOrdenado;
    hst_vectorInicial = (int *)malloc(tam * sizeof(int));
    hst_vectorOrdenado = (int *)malloc(tam * sizeof(int));
    // inicializacion de datos en el host
    srand((int)time(NULL));
    for (int i = 0; i < tam; i++)
    {
        hst_vectorInicial[i] = (rand() % 30) + 1;
    }
    int *dev_vectorInicial, *dev_vectorOrdenado;
    // reserva en el device
    cudaMalloc((void **)&dev_vectorInicial, tam * sizeof(int));
    // reserva en el device
    cudaMalloc((void **)&dev_vectorOrdenado, tam * sizeof(int));
    // lo pasamos a la GPU
    cudaMemcpy(dev_vectorInicial, hst_vectorInicial, tam * sizeof(int), cudaMemcpyHostToDevice);
    int bloque = 1;
    int numHilos = tam;
    int division = (tam / tammax);
    if (tam > tammax)
    {
        if (tam % tammax == 0)
        {
            bloque = division;
            numHilos = tammax;
        }
        else
        {
            bloque = 1 + division;
            numHilos = tammax;
        }
    }
    printf("> Numero de elementos que se van a ordenar (%d)\n\n", tam);
    printf("---------------------------------------------------\n");
    printf("Tiempo que tarda la GPU:\n");
    printf("---------------------------------------------------\n");
    printf("> Se han lanzado %d hilos en cada uno de los %d bloque/s (%d hilos)\n\n", numHilos, bloque, bloque * numHilos);
    //
    //
    dim3 Nbloques(bloque);
    dim3 hilosB(numHilos);
    // declaracion de eventos
    cudaEvent_t start;
    cudaEvent_t stop;
    // creacion de eventos
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // marca de inicio
    cudaEventRecord(start, 0);
    // codigo a temporizar en el device
    ordenarVectorPorRango<<<Nbloques, hilosB>>>(dev_vectorInicial, dev_vectorOrdenado, tam);
    // marca de final
    cudaEventRecord(stop, 0);
    // sincronizacion GPU-CPU
    cudaEventSynchronize(stop);
    // calculo del tiempo en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // impresion de resultados
    printf("> Tiempo de ejecucion GPU: %f ms\n", elapsedTime);
    // liberacion de recursos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ////////////////CPU///////////////////
    // declaraciones
    int *hst_B;
    // reserva en el host
    hst_B = (int *)malloc(tam * sizeof(int));
    printf("\n---------------------------------------------------\n");
    printf("Tiempo que tarda la CPU:\n");
    printf("\n");
    // TEMPORIZACION
    // Mediante eventos del sistema. Precision: fracciones de 1 ms.
    // La funcion setEvent() devuelve un evento de tiempo.
    // La funcion eventDiff() calcula la diferencia de tiempo (en milisegundos) entre dos eventos.
    event start1; // variable para almacenar el evento de tiempo inicial.
    event stop1;  // variable para almacenar el evento de tiempo final.
    double t_ms;
    double tiempo;
    // Iniciamos el contador
    setEvent(&start1); // marca de evento inicial
    // Multiplicacion de matrices
    for (int i = 0; i < tam; i++)
    {
        int rango = 0;
        for (int j = 0; j < tam; j++)
        {
            if (hst_vectorInicial[i] > hst_vectorInicial[j])
            {
                rango++;
            }
            else
            {
                if (hst_vectorInicial[i] == hst_vectorInicial[j] && i > j)
                {
                    rango++;
                }
            }
        }
        hst_B[rango] = hst_vectorInicial[i];
    }
    // Paramos el contador
    setEvent(&stop1); // marca de evento final
    // Intervalos de tiempo
    t_ms = eventDiff(&start1, &stop1); // diferencia de tiempo en ms
    tiempo = t_ms / 1000;              // tiempo en s
    printf("> Tiempo de ejecucion en la CPU: %f ms\n", t_ms);
    printf("\n> Ganancia GPU/CPU: %f\n", t_ms / elapsedTime);
    ///////////////////////////////////
    // Pasa el resultado al host
    cudaMemcpy(hst_vectorOrdenado, dev_vectorOrdenado, tam * sizeof(int), cudaMemcpyDeviceToHost);
    if (mostrar == 0)
    {
        // Se imprime el vector generado
        printf("\n---------------------------------------------------");
        printf("\nVector aleatorio:\n");
        for (int i = 0; i < tam; i++)
        {
            printf("%d ", hst_vectorInicial[i]);
        }
        // Se imprime el vector ordenado
        printf("\n---------------------------------------------------");
        printf("\nVector ordenado por la GPU:\n");
        for (int i = 0; i < tam; i++)
        {
            printf("%d ", hst_vectorOrdenado[i]);
        }
        printf("\n---------------------------------------------------");
        printf("\nVector ordenado por la CPU:\n");
        for (int i = 0; i < tam; i++)
        {
            printf("%d ", hst_B[i]);
        }
        printf("\n---------------------------------------------------");
    }
    printf("\n\n");
    printf("---------------------------------------------------\n");
    printf("Resultados:\n");
    printf("\n");
    printf("> Numero de elementos del vector = %d \n> [GPU: %f ms] \n> [CPU: %f ms] \n> [Ganancia GPU/CPU = %f] \n", tam, elapsedTime, t_ms, t_ms / elapsedTime);
    getchar();
    // Informacion del Sistema
    char infoName[1024];
    char infoUser[1024];
    DWORD longitud;
    GetComputerName(infoName, &longitud);
    GetUserName(infoUser, &longitud);
    time_t fecha;
    time(&fecha);
    printf("---------------------------------------------------\n\n");
    printf("Programa ejecutado el dia: %s", ctime(&fecha));
    printf("Maquina usada: %s\n", infoName);
    printf("Usuario: %s\n", infoUser);
    // SALIDA DEL PROGRAMA
    printf("<pulsa [INTRO] para finalizar>\n");
    getchar();
    return 0;
}
////////////////////////////////////////////////////////////////////
__host__ int propiedades_Device(int deviceID)
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
        // VOLTA(7.0) //TURING(7.5)
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
    }
    // presentacion de propiedades
    printf("---------------------------------------------------\n");
    printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
    printf("---------------------------------------------------\n");
    printf("> CUDA Toolkit \t: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    printf("> Arquitectura CUDA \t: %s\n", archName);
    printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
    printf("> No. de MultiProcesadores \t: %d\n", SM);
    printf("> No. de CUDA Cores (%dx%d) \t: %d\n", cudaCores, SM,
           cudaCores * SM);
    printf("> No. maximo de Hilos (por bloque)\t: %d\n", deviceProp.maxThreadsPerBlock);
    printf("> Memoria Global (total) \t: %zu MiB\n",
           deviceProp.totalGlobalMem / (1024 * 1024));
    printf("---------------------------------------------------\n");
    return deviceProp.maxThreadsPerBlock;
}