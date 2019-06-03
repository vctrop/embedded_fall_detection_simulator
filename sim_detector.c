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

#include "periodic_signals.h"

#define WINDOW_SIZE 1024
#define SERVER_PORT 49000

typedef struct gps_location{
    float latitude;
    float longitude;
} GPS; 

GPS current_location = {0.0, 0.0};

int flag_display = 0;                   // Enabled by server, send accelerometer data when true    

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
    
    make_periodic (, &info);
    while(1){
        // Load csv file circularly and update current_location;
        
        
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
}

void* write_socket(void* arg){
    // Information to be sent:
    // - Fall alert
    // - GPS location, when requested
    // - Accelerometer data, when enabled
}