/*
  Calculo de anomalia climatica paralelo (2025)
  Autor: Bastian Troncoso Retamales
  Sort -> Thrust en CPU
*/

#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <omp.h>
#include <unistd.h> // ( FUNCION unsleep() )
#include <math.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_segmented_radix_sort.cuh>

/*
  Contiene los parametros para cada caso de estudio
  FILE_NAME_PREC, FILE_NAME_TG, FILE_NAME_ANOMALY
  LAT, LON, TIME
  Añadir flag -DSMALL, -DMEDIUM, -DBIG
*/
#include "case_study.h" // 

int NUM_THREADS = 1;  
#define FLOAT_TOLERANCE 1e-5f
#define NC_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(EXIT_FAILURE);}

/*
  GLOBAL_QUEUE_SIZE
  Valores optimos con NC_FILL:
  BIG: 80000
  MEDIUM: 50000
  SMALL: 10000
*/
#define GLOBAL_QUEUE_SIZE 10000 // Por defecto con NC_NOFILL 

// Globales para depuración
volatile int tasks_enqueued = 0;
volatile int tasks_dequeued = 0;

// -----------------------------
// Estructuras de datos
// -----------------------------
typedef struct {
  short part_tg_val[TIME];
  float part_prec_val[TIME];
  float drought_code[TIME];
  short estres_termico[TIME];
  short estres_hidrico[TIME];
  short anomalia_climatica[TIME];
  short orderedTg[TIME];
  float orderedPrec[TIME];
} PointData;

typedef struct {
  short part_tg_val[TIME*LON];
  float part_prec_val[TIME*LON];
  float drought_code[TIME*LON];
  short estres_termico[TIME*LON];
  short estres_hidrico[TIME*LON];
  short anomalia_climatica_full[TIME*LON];
  short anomalia_climatica_empty[TIME*LON];
  short orderedTg[TIME*LON];
  float orderedPrec[TIME*LON];
} PointData_Batch;

typedef struct {
  int ncId;
  int varId;
  int lonId;
  int latId;
  int timeId;
} NetCDFFile;

// -----------------------------
// Funciones auxiliares
// -----------------------------
int calculateDroughtCode(PointData *data, double *days);
int encontrarPercentilShort(short *percentiles, short valor);
int encontrarPercentilFloat(float *percentiles, float valor);
int calculateEstresTermico(PointData *data);
int calculateEstresHidrico(PointData *data);
void calculateAnomaliaClimatica(PointData *data, unsigned long *not_valid);
void initNetCDFFiles(NetCDFFile *prec, NetCDFFile *tg, NetCDFFile *out, double *time, int NUM_THREADS);
double getDayLength(double day);


/* añadidas para poder ejecutar en Batch */
int calculateDroughtCode_Batch(short* part_tg_val, float* part_prec_val, float* drought_code, double *days);
void calculateAnomaliaClimatica_Batch(short* estres_termico, short* estres_hidrico, short* anomalia_climatica, unsigned long *not_valid);
int calculateEstresHidrico_Batch(float* orderedPrec, float* drought_code, short* estres_hidrico);
int calculateEstresTermico_Batch(short* part_tg_val, short* orderedTg, short* estres_termico);

__global__ void emptyKernel() {}

// -----------------------------
// MAIN
// -----------------------------
int main(int argc, char *argv[]) {

    if (argc != 2) {
      printf("Uso: %s <num_threads>\n", argv[0]);
      printf("Max threads: %d\n", omp_get_max_threads());
      return EXIT_FAILURE;
    }
    NUM_THREADS = atoi(argv[1]);
    NetCDFFile precFiles[NUM_THREADS], tgFiles[NUM_THREADS], outFile;
    double time[TIME];
    initNetCDFFiles(precFiles, tgFiles, &outFile, time, NUM_THREADS);
    unsigned long notCalculated = 0; // Total de celdas invalidas
    unsigned long total_not_valid = 0; // Total de valores invalidos

    double start_time, end_time;
    start_time = omp_get_wtime();

    /* changes to act in batch*/

    emptyKernel<<<1, 1>>>(); // GPU warmup

    PointData_Batch* data = (PointData_Batch*) malloc(sizeof(PointData_Batch));
    short* part_tg_val = (*data).part_tg_val;
    float* part_prec_val = (*data).part_prec_val;
    float* drought_code = (*data).drought_code;
    short* estres_termico = (*data).estres_termico;
    short* estres_hidrico = (*data).estres_hidrico;
    short* anomalia_climatica_aux;
    short* anomalia_climatica_full = (*data).anomalia_climatica_full;
    short* anomalia_climatica_empty = (*data).anomalia_climatica_empty;
    short* orderedTg = (*data).orderedTg;
    float* orderedPrec = (*data).orderedPrec;
    int retValues[LON];

    float* d_input;
    float* d_output;
    int h_offsets[LON+1];
    int* d_offsets;

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cudaStream_t stream;

    cudaStreamCreate(&stream);
    cudaMalloc(&d_input, LON*TIME*sizeof(float));
    cudaMalloc(&d_output, LON*TIME*sizeof(float));
    cudaMalloc(&d_offsets, (LON+1)*sizeof(int));

    for(int i = 0; i < (LON+1); i++) h_offsets[i] = TIME*i;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, 
                                            TIME*LON, LON, d_offsets, d_offsets+1, 0, sizeof(float)*8);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaMemcpy(d_offsets, h_offsets, (LON+1)*sizeof(int), cudaMemcpyHostToDevice);

    for(int j = 0; j < LAT; j++) {
      size_t start[3] = {(size_t)(j-1), 0, 0};
      size_t count[3] = {1, LON, TIME};

      #pragma omp parallel reduction(+:total_not_valid, notCalculated) num_threads(NUM_THREADS)
      {
        int threadId = omp_get_thread_num();

        #pragma omp single nowait
        {
          if(j != 0) {
            int retval = nc_put_vara_short(outFile.ncId, outFile.varId, start, count, anomalia_climatica_full);
            if (retval != NC_NOERR) {
              printf("Error al escribir en NetCDF en la celda (%d): %s\n", (j-1), nc_strerror(retval));
              exit(EXIT_FAILURE);
            }
          }
        }

        #pragma omp for nowait
        for (int k = 0; k < LON; k++) { // lectura de los datos + calculo del drought_code
          size_t read_start[3] = {(size_t)j, (size_t) k, 0};
          size_t read_count[3] = {1, 1, TIME};
          int ret1 = nc_get_vara_short(tgFiles[threadId].ncId, tgFiles[threadId].varId, read_start, read_count, part_tg_val+(k*TIME));
          int ret2 = nc_get_vara_float(precFiles[threadId].ncId, precFiles[threadId].varId, read_start, read_count, part_prec_val+(k*TIME));
          if (ret1 != NC_NOERR || ret2 != NC_NOERR) {
              printf("Error leyendo NetCDF en (%d, %d): %s\n", j, k, nc_strerror(ret1 != NC_NOERR ? ret1 : ret2));
              exit(EXIT_FAILURE);
          }
          int ret = calculateDroughtCode_Batch(part_tg_val+(k*TIME), part_prec_val+(k*TIME), drought_code+(k*TIME), time);
          retValues[k] = ret;
          if(ret) notCalculated++; 
        }

        #pragma omp single nowait
        {
          cudaMemcpyAsync(d_input, drought_code, LON*TIME*sizeof(float), cudaMemcpyHostToDevice, stream);
          cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, 
                                                TIME*LON, LON, d_offsets, d_offsets+1, 0, sizeof(float)*8, stream);
          cudaMemcpyAsync(orderedPrec, d_output, LON*TIME*sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
       
        #pragma omp for nowait
        for (int k = 0; k < LON; k++) {
          if(!(retValues[k])) {
            memcpy(orderedTg+(k*TIME), part_tg_val+(k*TIME), sizeof(short) * TIME);
            thrust::sort(thrust::host, orderedTg+(k*TIME), orderedTg+((k+1)*TIME));
            calculateEstresTermico_Batch(part_tg_val+(k*TIME), orderedTg+(k*TIME), estres_termico+(k*TIME));
          }
          else {
            for(int i=0; i<TIME; i++){
              anomalia_climatica_empty[(k*TIME)+i]=-1; /* esto se puede optimizar con memset */
            }
          }
        }

        #pragma omp single
        {
          cudaStreamSynchronize(stream);
        }
        #pragma omp barrier

        unsigned long local_not_valid = 0;

        #pragma omp for
        for (int k = 0; k < LON; k++) { // estres hidrico + anomalia climatica
          if(!(retValues[k])) {
            calculateEstresHidrico_Batch(orderedPrec+(k*TIME), drought_code+(k*TIME), estres_hidrico+(k*TIME));
            calculateAnomaliaClimatica_Batch(estres_termico+(k*TIME), estres_hidrico+(k*TIME), anomalia_climatica_empty+(k*TIME), &local_not_valid);
          }
        }
        total_not_valid += local_not_valid;

        anomalia_climatica_aux = anomalia_climatica_empty;
        anomalia_climatica_empty = anomalia_climatica_full;
        anomalia_climatica_full = anomalia_climatica_aux;
      }
    } // END LOOP LAT

    size_t start[3] = {(size_t) (LAT-1), 0, 0};
    size_t count[3] = {1, LON, TIME};
    int retval = nc_put_vara_short(outFile.ncId, outFile.varId, start, count, anomalia_climatica_full);
    if (retval != NC_NOERR) {
      printf("Error al escribir en NetCDF en la celda (%d): %s\n", LAT-1, nc_strerror(retval));
      exit(EXIT_FAILURE);
    }
    free(data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    cudaFree(d_temp_storage);

    total_not_valid += (notCalculated*TIME);
        
    /* end of changes*/
    
    end_time = omp_get_wtime();
    double seconds = end_time - start_time;
    printf("Cálculo finalizado. NotCalculated: %ld , Total no válidos: %ld\n", notCalculated, total_not_valid);
    printf("Duración total: %f segundos\n\n", seconds);

    return 0;
}

int calculateDroughtCode(PointData *data, double *days) {
    int i;
    int limite = TIME * 0.3;
    int invalid, totalInvalid = 0;
    float prevDroughtCode = 0;
    float efRainfall, moisture, prevMoisture, evapotranspiration;
    for (i = 1; i < TIME; i++) {
        invalid = 0;

        if (data->part_prec_val[i] < 0.0 || data->part_prec_val[i] > 300.0) {
            data->part_prec_val[i] = 0;
            invalid = 1;
        }

        if (data->part_tg_val[i] < -8000 || data->part_tg_val[i] > 8000) {
            data->part_tg_val[i] = 1040;
            invalid = 1;
        }

        if (invalid) totalInvalid++;

        if (data->part_prec_val[i] > 2.8) {
            efRainfall = 0.86 * data->part_prec_val[i] - 1.27;
            prevMoisture = 800 * exp(-data->drought_code[i] / 400.0);
            moisture = prevMoisture + 3.937 * efRainfall;
            prevDroughtCode = 400 * log(800.0 / moisture);
            if (prevDroughtCode < 0) prevDroughtCode = 0;
        }

        evapotranspiration = 0.36 * (data->part_tg_val[i] + 2.0) + getDayLength(days[i]);
        if (evapotranspiration < 0) evapotranspiration = 0;
        data->drought_code[i] = prevDroughtCode + 0.5 * evapotranspiration;
    }
    return (totalInvalid > limite) ? 1 : 0;
}

int calculateDroughtCode_Batch(short* part_tg_val, float* part_prec_val, float* drought_code, double *days) {
    int i;
    int limite = TIME * 0.3;
    int invalid, totalInvalid = 0;
    float prevDroughtCode = 0;
    float efRainfall, moisture, prevMoisture, evapotranspiration;
    for (i = 1; i < TIME; i++) {
        invalid = 0;

        if (part_prec_val[i] < 0.0 || part_prec_val[i] > 300.0) {
            part_prec_val[i] = 0;
            invalid = 1;
        }

        if (part_tg_val[i] < -8000 || part_tg_val[i] > 8000) {
            part_tg_val[i] = 1040;
            invalid = 1;
        }

        if (invalid) totalInvalid++;

        if (part_prec_val[i] > 2.8) {
            efRainfall = 0.86 * part_prec_val[i] - 1.27;
            prevMoisture = 800 * exp(-drought_code[i] / 400.0);
            moisture = prevMoisture + 3.937 * efRainfall;
            prevDroughtCode = 400 * log(800.0 / moisture);
            if (prevDroughtCode < 0) prevDroughtCode = 0;
        }

        evapotranspiration = 0.36 * (part_tg_val[i] + 2.0) + getDayLength(days[i]);
        if (evapotranspiration < 0) evapotranspiration = 0;
        drought_code[i] = prevDroughtCode + 0.5 * evapotranspiration;
    }
    return (totalInvalid > limite) ? 1 : 0;
}

int encontrarPercentilShort(short *percentiles, short valor) {
    int left = 0, right = 97, result = 98;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (valor <= percentiles[mid]) {
            result = mid + 1;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

int encontrarPercentilFloat(float *percentiles, float valor) {
    int left = 0, right = 97, result = 98;
    while (left <= right) {
        int mid = (left + right) / 2;
        float diff = valor - percentiles[mid];
        if (diff < -FLOAT_TOLERANCE) {
            result = mid + 1;
            right = mid - 1;
        } else if (fabsf(diff) < FLOAT_TOLERANCE) {
            return mid + 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

int calculateEstresTermico(PointData *data) {
    short percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = data->orderedTg[i * tamano_percentil];

    for (int i = 0; i < 7; i++)
        data->estres_termico[i] = -1;

    for (int i = 7; i < TIME; i++) {
        int suma = 0;
        for (int k = 0; k < 8; k++)
            suma += data->part_tg_val[i - k];
        short media = (short)(suma / 8);
        data->estres_termico[i] = encontrarPercentilShort(percentiles, media);
    }
    return 0;
}

int calculateEstresTermico_Batch(short* part_tg_val, short* orderedTg, short* estres_termico) {
    short percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = orderedTg[i * tamano_percentil];

    for (int i = 0; i < 7; i++)
        estres_termico[i] = -1;

    for (int i = 7; i < TIME; i++) {
        int suma = 0;
        for (int k = 0; k < 8; k++)
            suma += part_tg_val[i - k];
        short media = (short)(suma / 8);
        estres_termico[i] = encontrarPercentilShort(percentiles, media);
    }
    return 0;
}

int calculateEstresHidrico(PointData *data) {
    float percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = data->orderedPrec[i * tamano_percentil];

    for (int i = 0; i < TIME; i++)
        data->estres_hidrico[i] = encontrarPercentilFloat(percentiles, data->drought_code[i]);

    return 0;
}

int calculateEstresHidrico_Batch(float* orderedPrec, float* drought_code, short* estres_hidrico) {
    float percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = orderedPrec[i * tamano_percentil];

    for (int i = 0; i < TIME; i++)
        estres_hidrico[i] = encontrarPercentilFloat(percentiles, drought_code[i]);

    return 0;
}

void calculateAnomaliaClimatica(PointData *data, unsigned long *not_valid) {
  for (int i = 0; i < TIME; i++) {
    if (data->estres_termico[i] < 1 || data->estres_termico[i] > 99 ||
        data->estres_hidrico[i] < 1 || data->estres_hidrico[i] > 99) {
        data->anomalia_climatica[i] = -1;
      (*not_valid)++;
    }
    else {
      data->anomalia_climatica[i] = (data->estres_termico[i] + data->estres_hidrico[i]) / 2;
    }
  }
}

void calculateAnomaliaClimatica_Batch(short* estres_termico, short* estres_hidrico, short* anomalia_climatica, unsigned long *not_valid) {
  for (int i = 0; i < TIME; i++) {
    if (estres_termico[i] < 1 || estres_termico[i] > 99 ||
        estres_hidrico[i] < 1 || estres_hidrico[i] > 99) {
        anomalia_climatica[i] = -1;
      (*not_valid)++;
    }
    else {
      anomalia_climatica[i] = (estres_termico[i] + estres_hidrico[i]) / 2;
    }
  }
}

void initNetCDFFiles(NetCDFFile *prec, NetCDFFile *tg, NetCDFFile *out, double *time, int NUM_THREADS) {
    int retval;
    double latitudes[LAT], longitudes[LON];

    // Apertura de archivos para todos los hilos
    for (int i = 0; i < NUM_THREADS; i++) {
      if ((retval = nc_open(FILE_NAME_PREC, NC_NOWRITE, &prec[i].ncId))) NC_ERR(retval);
      if ((retval = nc_open(FILE_NAME_TG, NC_NOWRITE, &tg[i].ncId)))     NC_ERR(retval);

      if ((retval = nc_inq_varid(prec[i].ncId, "rr", &prec[i].varId)))   NC_ERR(retval);
      if ((retval = nc_inq_varid(tg[i].ncId, "tg", &tg[i].varId)))       NC_ERR(retval);
    }

    // Obtenemos timeId, lonId, latId desde el primer archivo de precipitación
    if ((retval = nc_inq_dimid(prec[0].ncId, "longitude", &prec[0].lonId))) NC_ERR(retval);
    if ((retval = nc_inq_dimid(prec[0].ncId, "latitude", &prec[0].latId)))  NC_ERR(retval);
    if ((retval = nc_inq_dimid(prec[0].ncId, "time", &prec[0].timeId)))     NC_ERR(retval);

    // Leemos valores de coordenadas y tiempo
    if ((retval = nc_get_var_double(prec[0].ncId, prec[0].timeId, time)))      NC_ERR(retval);
    if ((retval = nc_get_var_double(prec[0].ncId, prec[0].lonId, longitudes))) NC_ERR(retval);
    if ((retval = nc_get_var_double(prec[0].ncId, prec[0].latId, latitudes)))  NC_ERR(retval);

    // Crear archivo de salida
    if ((retval = nc_create(FILE_NAME_ANOMALY, NC_NETCDF4, &out->ncId))) NC_ERR(retval);

    // Definir dimensiones
    if ((retval = nc_def_dim(out->ncId, "latitude", LAT, &out->latId)))   NC_ERR(retval);
    if ((retval = nc_def_dim(out->ncId, "longitude", LON, &out->lonId))) NC_ERR(retval);
    if ((retval = nc_def_dim(out->ncId, "time", TIME, &out->timeId)))    NC_ERR(retval);

    // Definir variables de coordenadas
    int latVarId, lonVarId, timeVarId;
    if ((retval = nc_def_var(out->ncId, "latitude", NC_DOUBLE, 1, &out->latId, &latVarId))) NC_ERR(retval);
    if ((retval = nc_def_var(out->ncId, "longitude", NC_DOUBLE, 1, &out->lonId, &lonVarId))) NC_ERR(retval);
    if ((retval = nc_def_var(out->ncId, "time", NC_DOUBLE, 1, &out->timeId, &timeVarId))) NC_ERR(retval);

    // Definir variable de salida
    int dims[3] = {out->latId, out->lonId, out->timeId};
    if ((retval = nc_def_var(out->ncId, "anomaliaClimatica", NC_SHORT, 3, dims, &out->varId))) NC_ERR(retval);

    /* 
    Desactiva el pre-relleno con _FillValue.
    Por defecto, NetCDF inicializa toda la variable con un valor por defecto (_FillValue),
    // lo cual genera una enorme latencia en la primera escritura. Con NC_NOFILL, 
    // se evita ese pre-relleno y los datos se escriben directamente al realizar nc_put_vara().
    */
    printf("Using NC_NOFILL\n");
    if ((retval = nc_def_var_fill(out->ncId, out->varId, NC_NOFILL, NULL))) NC_ERR(retval);

    /* 
    Desactiva la compresión y el filtro "shuffle".
    NetCDF-4 permite comprimir los datos (deflate) y reordenar bytes (shuffle),
    pero esto obliga al uso de layout CHUNKED y ralentiza drásticamente las escrituras.
    Con estos tres ceros (shuffle=0, deflate=0, deflate_level=0), los datos se almacenan sin compresión.
    */
    //if ((retval = nc_def_var_deflate(out->ncId, out->varId, 0, 0, 0))) NC_ERR(retval);

    /* 
    Define el layout de almacenamiento como CONTIGUOUS (bloque lineal).
    En vez de dividir la variable en miles de "chunks" (bloques pequeños),
    CONTIGUOUS la guarda secuencialmente en disco, reduciendo metadatos y
    mejorando la velocidad de escritura en patrones secuenciales (como tu caso LAT×LON×TIME).
    */
    //if ((retval = nc_def_var_chunking(out->ncId, out->varId, NC_CONTIGUOUS, NULL))) NC_ERR(retval);

    // Finalizar definición
    if ((retval = nc_enddef(out->ncId))) NC_ERR(retval);

    // Escribir coordenadas
    if ((retval = nc_put_var_double(out->ncId, latVarId, latitudes)))  NC_ERR(retval);
    if ((retval = nc_put_var_double(out->ncId, lonVarId, longitudes))) NC_ERR(retval);
    if ((retval = nc_put_var_double(out->ncId, timeVarId, time)))      NC_ERR(retval);
}

double getDayLength(double day){
  int dayYear = (int) day%365;
  if(dayYear < 90)
    return -1.6;
  if(dayYear<120)
    return 0.9;
  if(dayYear<151)
    return 3.8;
  if(dayYear<181)
    return 5.8;
  if(dayYear<212)
    return 6.4;
  if(dayYear<243)
    return 5.0;
  if(dayYear<273)
    return 2.4;
  if(dayYear<304)
    return 0.4;
  //if(dayYear<365)
  return -1.6;  
}
