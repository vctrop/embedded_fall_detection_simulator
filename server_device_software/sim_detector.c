#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <signal.h>
#include <errno.h>
#include <math.h>
#include "periodic_signals.h"

// General definitions
#define NUM_THREADS 5
// Server definitions
#define SERVER_ADDRESS "127.0.0.1"
// Request definitions
#define RQST_DATA_ENABLE    1
#define RQST_DATA_DISABLE   -1
#define RQST_LOCATION	    2	
// Data definitions
#define SOCK_SEND_BSIZE 1698			// 3 + 17*2 + 11*151
#define GPS_BSIZE 80
#define ACQUISITION_BSIZE 90
#define WINDOW_SIZE 151
#define NUM_FEATURES 3
// Periodic definitions
#define GPS_PERIOD		5000000		// 5s
#define SENSOR_PERIOD		  20000		// 0.02s
#define DETECTOR_PERIOD		3020000		// 3.02 s


// Application
typedef struct gps_location{
    float latitude;
    float longitude;
} GPS;

// Global variables
int sockfd;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

GPS current_location = {0.0, 0.0};
char flag_display = 0;                     // Set by server, send accelerometer data when true    
char flag_location = 0;                    // Set by server, send location
char flag_fall_detected = 0; 
short int send_socket = 0;                 // Set by buffer crossing, fall detection or gps request, cleared when socket is sent
                          
float data_buffer[2*WINDOW_SIZE];               // Circular double buffer to keep sensor data
int sb_index = 0;                               // Sensor buffer index
short int buf_dirty_half = 1;                   // Indicates which buffer half is dirty 


// Threads
void* read_gps(void* arg);                      // GPS acquisition thread
void* read_accelerometer(void* arg);            // Accelerometer acquisition thread
void* estimate_fall(void* arg);                 // Thread to apply signal processing and classification
void* read_socket(void* arg);                   // 
void* write_socket(void* arg);                  // Socket communication threads

int main(int argc, char *argv[]) {
    int i, portno;
    sigset_t alarm_sig;
    void* threads_vec[NUM_THREADS];
    pthread_t temp_thread;
    struct sockaddr_in serv_addr;
    cpu_set_t cpu_set;
    
    if (argc < 2) {
        perror("Error, port undefined:");
        exit(-1);
    }
    portno = atoi(argv[1]);
    
    // Block all RT signals, so they can be used for the timers (this has to be done before any threads are created)
    sigemptyset(&alarm_sig);
    for (i= SIGRTMIN; i <= SIGRTMAX; i++)
        sigaddset(&alarm_sig, i);
    sigprocmask(SIG_BLOCK, &alarm_sig, 0);
    
    // Stabilish connection with server
    sockfd = socket(AF_INET, SOCK_STREAM, 0);                    // Create connection-oriented socket descriptor using IPv4 protocol
    if (sockfd < 0){
        perror("Error creating socket:");
        exit(-1);
    }
    memset((char*) &serv_addr, 0, sizeof(serv_addr));            // Clear server address structure
    serv_addr.sin_family = AF_INET;                                 // Stabilish address family
    inet_aton(SERVER_ADDRESS, &serv_addr.sin_addr);                 // Convert server address from IPv4 numbers-and-dots to network byte order
    serv_addr.sin_port = htons(portno);                        
    while(connect(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0)    // Only advance when connection succeed
        perror("Error when stabilishing connection (trying again):");
    
    // Clear data buffer
    memset(data_buffer, 0, 2*WINDOW_SIZE*(sizeof(float)));        // Clear data buffer
    
    // Initiate threads
    CPU_ZERO(&cpu_set);                                             //
    CPU_SET(0, &cpu_set);                                           // Only CPU 0 is used by the threads        
    threads_vec[0] = read_gps;
    threads_vec[1] = read_accelerometer;
    threads_vec[2] = estimate_fall;
    threads_vec[3] = read_socket;
    threads_vec[4] = write_socket;
    for (i=0; i<NUM_THREADS; i++){
        if(pthread_create(&temp_thread, 0, threads_vec[i], 0)){
            perror("Error on thread creation:");
            exit(-1);
        }
        pthread_setaffinity_np(temp_thread, sizeof(cpu_set_t), &cpu_set);
    }
    
    while(1){
        
    }
    
    return 0;
}

void* read_gps(void* arg){
    struct periodic_info info;
    char filename[] = "go_track_trackspoints.csv";
    char gps_buffer[GPS_BSIZE];
    FILE *fd;
    char *field;
    
    fd = fopen(filename, "r");
    if (!fd){
        perror("Error opening GPS csv:");
        exit(-1);
    }
    
    make_periodic (GPS_PERIOD, &info);
    while(1){
        // Load csv file circularly and update current_location;
        if (!fgets(gps_buffer, GPS_BSIZE, fd))
            rewind(fd);
        
        field = strtok(gps_buffer, ",");                // ignore first field
        
        field = strtok(NULL, ",");
        current_location.latitude = atof(field);    // get latitude
        field = strtok(NULL, ",");
        current_location.longitude = atof(field);   // get longitude
        
        printf("Current location = %f, %f\n", current_location.latitude, current_location.longitude);
        wait_period (&info);
    }
}

void* read_accelerometer(void* arg){
    struct periodic_info info;
    char filename[] = "3d_partial_data.csv";
    char acquisition_buffer[ACQUISITION_BSIZE];
    FILE *fd;
    char *field;
    double axis_h, axis_j, axis_k;
    
    fd = fopen(filename, "r");
    if (!fd){
        perror("Error opening sensor csv:");
        exit(-1);
    }
    
    make_periodic (SENSOR_PERIOD, &info);
    while(1){
        // Load data from csv and insert j dimension in circular buffer
        if (!fgets(acquisition_buffer, ACQUISITION_BSIZE, fd))
            rewind(fd);
        
        field = strtok(acquisition_buffer, ",");
        axis_h = atof(field);
        field = strtok(NULL, ",");
        axis_j = atof(field);
        field = strtok(NULL, ",");
        axis_k = atof(field);
        
        // Store dimension j in circular buffer
        pthread_mutex_lock(&mutex);
	        data_buffer[sb_index] = (float) axis_j;
	        sb_index = (sb_index+1)%(2*WINDOW_SIZE);
	        if((sb_index == 0) || (sb_index == 151)){
	            if (sb_index == 0)
	                buf_dirty_half = 0;
	            else
	                buf_dirty_half = 1;
	            
	            if (flag_display == 1){
	                send_socket = 1;
	                pthread_cond_signal(&cond);
	            }
        	}
	pthread_mutex_unlock(&mutex);	
        
        //printf("GOT ACCELEROMETER = %f\n", axis_j);
        wait_period (&info);
        
    }
}

void* estimate_fall(void* arg){
    struct periodic_info info;
    int i, buffer_i;                     
    float dot_product;
    float sum, average;
    int zero_cross_count;
    
    make_periodic (DETECTOR_PERIOD, &info);
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
                zero_cross_count++;
        }
        // Predict
        dot_product = (-0.00740976)*average + (-0.09174859)*zero_cross_count;
        if (dot_product > 0){
            flag_fall_detected = 1;
            send_socket = 1;
            pthread_cond_signal(&cond);
        }
        else
            flag_fall_detected = 0;
            
        wait_period (&info);
    }
}

void* read_socket(void* arg){
    // Server requests to be decoded:
    // - Get GPS location
    // - Enable accelerometer data display
    // - Disable accelerometer data display
    char request_buffer[1];
    int num_bytes;
    
    while (1){
        memset(request_buffer, 0, sizeof(char));             // clear buffer
        num_bytes = read(sockfd, request_buffer, 1);        // receive data from socket
        if(num_bytes < 0){
            perror("Error on socket reading:");
            exit(-1);
        }
        //printf("Received request = %d\n", (int) request_buffer[0]);
        if(request_buffer[0]){
	        pthread_mutex_lock(&mutex);
	        switch(request_buffer[0]){
	            case RQST_LOCATION:                             // Check for location request
	            	printf("Received location request\n");
	                flag_location = 1;
	                send_socket = 1;
	                pthread_cond_signal(&cond);
	                break;
	            
	            case RQST_DATA_ENABLE:                          // Check for data display enable request
	                printf("Received data display enable request\n");
	                flag_display = 1;
	                break;
	            
	            case RQST_DATA_DISABLE:                         // Check for data display disable request
	                printf("Received data display disable request\n");
	               	flag_display = 0;
	                break;
        	}   
        	pthread_mutex_unlock(&mutex);
	}	
    }  
}

void* write_socket(void* arg){
    // Information to be sent:
    // - Fall alert
    // - GPS location, when requested
    // - Accelerometer data, when enabled
    int i, char_i, buffer_i, data_buffer_i;
    int ret;
    char coordinate_string[18];
    char data_string[11];
    char info_buffer[SOCK_SEND_BSIZE];		// Fall, [loc available, x, y] [data available, DATA]
    
    while (1){
        pthread_mutex_lock(&mutex);
            while(send_socket == 0)                     //
                pthread_cond_wait(&cond, &mutex);       // Wait on condition variable send_socket
            send_socket = 0;
            printf("Sending socket\n");
            info_buffer[0] = flag_fall_detected;        //
            info_buffer[1] = flag_location;             //
            info_buffer[2] = flag_display;              // Put flags in info buffer
            if (flag_fall_detected)
                flag_fall_detected = 0;           
            
            if(flag_location){
                // Fulfill info_buffer with location data (info_buffer[3:20[ <- latitude, info_buffer[20, 37[ <- longitude)
                flag_location = 0;
                buffer_i = 3;
                printf("Buffering GPS data to send\n");
                ret = (int) gcvt(current_location.latitude, 15, coordinate_string);         // Convert latitude to string using 15 digits
                printf("latitude: ");
                puts(coordinate_string);
                
                if(ret == 0){
                    perror("Error when converting float to string:");
                    exit(-1);
                }
                for(char_i=0; char_i<17; char_i++, buffer_i++){
                    info_buffer[buffer_i] = coordinate_string[char_i];
                    //printf("%c", info_buffer[buffer_i]);
                }
                
                ret = (int) gcvt(current_location.longitude, 15, coordinate_string);        // Convert longitude to string using 15 digits
                printf("longitude: ");
                puts(coordinate_string);
                if(ret == 0){
                    perror("Error when converting float to string:");
                    exit(-1);
                }
                for(char_i=0; char_i<17; char_i++, buffer_i++){
                    info_buffer[buffer_i] = coordinate_string[char_i];
            	    //printf("%c", info_buffer[buffer_i]);
            	}
            }

            if(flag_display){
                // Fulfill info_buffer with accelerometer data, info_buffer[37:1547[
                buffer_i = 37;
                printf("Buffering sensor data to send\n");
                for(i = 0; i < WINDOW_SIZE; i++){
                    buffer_i = 37 + i * 11;
                    data_buffer_i = i + (1 - buf_dirty_half) * WINDOW_SIZE;
                    ret = (int) gcvt(data_buffer[data_buffer_i], 8, data_string);
                    if(ret == 0){
                        perror("Error when converting float to string:");
                        exit(-1);
                    }
                    for(char_i = 0; char_i<11; char_i++, buffer_i++)
                        info_buffer[buffer_i] = data_string[char_i];
		    info_buffer[buffer_i-1] = ' ';  			// put space at the end of each number                           
                }
            }
        
        ret = send(sockfd, info_buffer, SOCK_SEND_BSIZE, 0);
        printf("Buffer sent to socket\n");
        if (ret < 0){
            perror("Error on socket writing:");
            exit(-1);
        }
        pthread_mutex_unlock(&mutex);
        
    }
}
