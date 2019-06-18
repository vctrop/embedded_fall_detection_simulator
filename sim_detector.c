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
#include "periodic_signals.h"

#define WINDOW_SIZE 1024
#define SERVER_PORT 49000
#define GPS_BSIZE 80

#define RQST_LOCATION	    1	
#define RQST_DATA_ENABLE    1
#define RQST_DATA_DISABLE   -1	

#define GPS_PERIOD
#define SENSOR_PERIOD
#define DETECTOR_PERIOD


typedef struct gps_location{
    float latitude;
    float longitude;
} GPS; 

GPS current_location = {0.0, 0.0};

int flag_display = 0;                   // Enabled by server, send accelerometer data when true    
int flag_location = 0;

char fall_detected = 0;
int sockfd;

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
        if (!fgets(buffer, GPS_BSIZE, fd))
            rewind(fp);
        
        field = strtok(buffer, ",");                // ignore first field
        
        field = strtok(NULL, ",");
        current_location.latitude = atof(field);    // get latitude
        field = strtok(NULL, ",");
        current_location.longitude = atof(field);   // get latitude
        
        wait_period (&info);
    }
}

void* read_accelerometer(void* arg){
    struct periodic_info info;
    
    make_periodic (, &info);
    while(1){
        // Load data from csv to a circular buffer at a sampling frequency
        
        
        wait_period (&info);
    }
}

void* estimate_fall(void* arg){
    struct periodic_info info;
    
    make_periodic (, &info);
    while(1){
        // Read window from buffer, extract features and apply decision function
        
        
        wait_period (&info);
    }
}

void* read_socket(void* arg){
    // Server requests to be decoded:
    // - Get GPS location
    // - Enable accelerometer data display
    // - Disable accelerometer data display
    char request_buffer[2];
    
    
    while (1){
        // clear buffer
        // receive socket
        // check field 1 for location request and 2 for data transfer enable
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