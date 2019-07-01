#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <math.h>
#include "periodic_signals.h"

#define SERVER_PORT 49000
#define GPS_BSIZE 80
#define ACQUISITION_BSIZE 90
#define WINDOW_SIZE 151
#define NUM_FEATURES 3

#define RQST_DATA_ENABLE    1
#define RQST_DATA_DISABLE   -1
#define RQST_LOCATION	    2	

#define GPS_PERIOD
#define SENSOR_PERIOD
#define DETECTOR_PERIOD

typedef struct gps_location{
    float latitude;
    float longitude;
} GPS;

// Global variables
GPS current_location = {0.0, 0.0};
short int flag_display = 0;                     // Set by server, send accelerometer data when true    
short int flag_location = 0;                    // Set by server, send location
char fall_detected = 0; 
int sockfd;                                     
float data_buffer[2*WINDOW_SIZE];             // Circular double buffer to keep sensor data
int sb_index = 0;                               // Sensor buffer index
short int buf_dirty_half = 1;                   // Indicates which buffer half is dirty 

// Sensor threads
void* read_gps(void* arg);
void* read_accelerometer(void* arg);

// Signal processing and classification thread
void* estimate_fall(void* arg);

// Socket communication threads
void* read_socket(void* arg);
void* write_socket(void* arg);

int main(int argc, char *argv[]) {
    int i;
    sigset_t alarm_sig;
    
    // Initially block all RT signals
    sigemptyset(&alarm_sig);
    for (i= SIGRTMIN; i <= SIGRTMAX; i++){
        sigaddset(&alarm_sig, i);
    sigprocmask(SIG_BLOCK, &alarm_sig, NULL);
    
    memset(&data_buffer, 0, 2*WINDOW_SIZE*(sizeof float));
    
    return 0;
}

void* read_gps(void* arg){
    struct periodic_info info;
    char filename[] = "go_track_trackspoints.csv";
    char gps_buffer[GPS_BSIZE];
    FILE *fd;
    char *field;
    
    fd = fopen(filename, 'r');
    if (fd == NULL){
        perror("Error opening GPS csv:");
        exit(-1);
    }
    
    make_periodic (, &info);
    while(1){
        // Load csv file circularly and update current_location;
        if (!fgets(gps_buffer, GPS_BSIZE, fd))
            rewind(fd);
        
        field = strtok(gps_buffer, ",");                // ignore first field
        
        field = strtok(NULL, ",");
        current_location.latitude = atof(field);    // get latitude
        field = strtok(NULL, ",");
        current_location.longitude = atof(field);   // get latitude
        
        wait_period (&info);
    }
}

void* read_accelerometer(void* arg){
    struct periodic_info info;
    char filename[] = "3d_full_data.csv";
    char acquisition_buffer[ACQUISITION_BSIZE];
    FILE *fd;
    char *field;
    double axis_h, axis_j, axis_k;
    
    fd = fopen(filename, 'r');
    if (fd == NULL){
        perror("Error opening sensor csv:");
        exit(-1);
    }
    
    make_periodic (, &info);
    while(1){
        // Load data from csv and insert j dimension in circular buffer
        if (!fgets(acquisiton_buffer, ACQUISITION_BSIZE, fd))
            rewind(fd);
        
        field = strtok(acquisiton_buffer, ",");
        axis_h = atof(field);
        field = strtok(NULL, ",");
        axis_j = atof(field);
        field = strtok(NULL, ",");
        axis_k = atof(field);
        // signal_mag = sqrt(pow(axis_h,2) + pow(axis_j,2) + pow(axis_k,2));
        
        // Store magnitude in circular buffer
        data_buffer[sb_index] = (float) axis_j;
        sb_index = (sb_index+1)%(2*WINDOW_SIZE);
        if (sb_index == 0)
            buf_dirty_half = 0;
        else if (sb_index == 151)
            buf_dirty_half = 1;
        
        wait_period (&info);
    }
}

void* estimate_fall(void* arg){
    struct periodic_info info;
    int window_begin, window_end;
    int i, buffer_i;                     
    float dot_product;
    float sum, average, std;
    int zero_cross_count;
    
    make_periodic (, &info);
    while(1){
        // Extract features (average, zero_cross_count)
        sum = 0.0;
        for(i = 0; i < WINDOW_SIZE; i++){
            buffer_i = i + (1 - buf_dirty_half) * WINDOW_SIZE;
            sum += data_buffer[buffer_i];
        }
        average = sum/WINDOW_SIZE;
        
        zero_cross_count = 0;
        for(i = 0; i < WINDOW_SIZE; i++){
            buffer_i = i + (1 - buf_dirty_half) * WINDOW_SIZE;
            if((data_buffer[buffer_i]-average)*(data_buffer[(buffer_i-1)%WINDOW_SIZE]-average) < 0)     // if sign has changed
        }       zero_cross_count++;
        
        // Predict
        dot_product = (-0.00740976)*average + (-0.09174859)*zero_cross_count;
        if dot_product > 0:
            fall_detected = 1;
        else
            fall_detected = 0;
            
        wait_period (&info);
    }
}

void* read_socket(void* arg){
    // Server requests to be decoded:
    // - Get GPS location
    // - Enable accelerometer data display
    // - Disable accelerometer data display
    char request_buffer[1];
    
    while (1){
        // clear buffer
        // receive socket
        // check field 1 for location request and 2 for data transfer enable/disable
    }	// set flags
}

void* write_socket(void* arg){
    // Information to be sent:
    // - Fall alert
    // - GPS location, when requested
    // - Accelerometer data, when enabled
    char info_buffer[]		// Fall, [loc available, x, y] [data available, DATA]
    
    while (1){
        // put fall state in buffer, clear fall state
        // put flag_location in buffer, clear flag_location
        // put x, y in buffer
        // put data_av in buffer, put data in buffer
        // send buffer and check for errors
    }
}