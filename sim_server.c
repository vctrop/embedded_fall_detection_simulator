#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

#include "linked_list.h"

#define SERVER_PORT 49000
#define MAX_DEVICES 50

#define RQST_DATA_ENABLE    1
#define RQST_DATA_DISABLE   -1
#define RQST_LOCATION	    2	

pthread_mutex_t mutex_accept = PTHREAD_MUTEX_INITIALIZER;

Node* connections = list_create();
int head_id = 0;

// A thread is assigned to each connected device
void* data_handler(void* id);
void* request_handler(void* id);
int connected_devices = 0;

int main(int argc, char *argv[]) {
    int serv_sockfd, new_sockfd, port_num, temp_ret;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t cli_addr_len;
    pthread_t thread;
    
    // Unbound server socket with protocol and type definition 
    serv_sockfd = socket(AF_INET, SOCK_STREAM, NULL);
    if (serv_sockfd < 0){
        perror("Socket creation error:");
        exit(-1);
    }
    
    // Configure socket address and bind it to serv_sockfd
    memset(&serv_addr, 0, sizeof serv_addr);
    port_num = htons(SERVER_PORT);              // Port number in network byte order
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = port_num;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    temp_ret = bind(serv_sockfd, (struct sockaddr*) &serv_addr, sizeof serv_addr);
    if (temp_ret < 0){
        perror("Bind error:");
        exit(-1);
    }
    
    // Listen for socket connections
    listen(serv_sockfd, MAX_DEVICES);
    cli_addr_len = sizeof cli_addr;
    while (1){
        // Extract connection from pending queue and create a new socket of the same type as serv_sockfd
        new_sockfd = accept(serv_sockfd, (struct sockaddr*) &cli_addr, &cli_addr_len);
        if (new_sockfd < 0){
            perror("Accept error:");
            exit(-1);
        }
        
        connected_devices++;
        
        // Insert a node in the connections list and create a thread to handle the device connection
        pthread_mutex_lock(&mutex_accept);
            connections = list_insert(connections, head_id, new_sockfd);
            pthread_create(&thread, NULL, data_handler, (void*) head_id);
            pthread_create(&thread, NULL, request_handler, (void*) head_id);
            head_id++;
        pthread_mutex_unlock(&mutex_accept);
    }   
    
    return 0; 
}

void* data_handler(void* id){
    // Data to be received:
    // - Fall alert (sporadic)
    // - GPS location (once, when requested)
    // - Accelerometer data (continuously, when enabled)

    int data_id = (int) id;
    
    while (1){
        memset(&, 0, sizeof );
        
    }
    
}

void* request_handler(void* id){
    // Requests to be sent:
    // - Get GPS location                       (l) 
    // - Enable accelerometer data display      (e)
    // - Disable accelerometer data display     (d)
    
    int req_id = (int) id;
    int request;
    char request_buffer[1];
    char c;
    Nodo* temp_nodo;
    
    while(1){
        // Get request from stdin
        switch(getchar()){
            case 'l':
                request = RQST_LOCATION;
                break;
            case 'e':
                request = RQST_DATA_ENABLE;
                break;
            case 'd':
                request = RQST_DATA_DISABLE;
                break;
            default:
                request = 0;
        }
        request_buffer[0] = (char) request;
        
        // Send request via socket
        temp_nodo = list_search(connections, req_id);
        if (write(temp_nodo->newsockfd, request_buffer, 1) < 0){
            perror("Socket writing error:");
            exit(-1);
        }
        request_buffer[0] = 0;

    }
}