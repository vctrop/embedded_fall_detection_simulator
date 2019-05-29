/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <netdb.h> 
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define WINDOW_SIZE 1024
#define SERVER_PORT 49000

struct geo_location{
    float latitude;
    float longitude;
};

struct geo_location current_location = {0.0, 0.0};

// Sensor threads
void* read_gps(void* arg);
void* read_accelerometer(void* arg);

// Signal processing and classification thread
void* estimate_fall(void* arg);

// Socket communication threads
void* read_socket(void* arg);
void* write_socket(void* arg);

int main(int argc, char *argv[]) {

    return 0; 
}

void* read_gps(void* arg){
    while(1){
        // Load csv file circularly and update current_location;
    }
}

void* read_accelerometer(void* arg){
    while(1){
        // Load data from csv to a circular buffer at a sampling frequency
    }
}

void* estimate_fall(void* arg){
    while(1){
        // Read window from buffer, extract features and apply decision function
        // If fall is detected, wake up socket writer thread
    }
}

void* read_socket(void* arg){
    // Decode server requests
}

void* write_socket(void* arg){
    // Send fall alert or info by server request
}