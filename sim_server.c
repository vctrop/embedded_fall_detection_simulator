#undef putchar

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

#include "linked_list.h"

#define SOCK_SEND_BSIZE 1547
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
    char info_buffer[SOCK_SEND_BSIZE];
    char latitude_string[18];
    char longitude_string[18];
    char data_string[11];
    int handler_id = (int) id;
    int n;
    int i, char_i, buffer_i, data_buffer_i;
    Nodo* temp_nodo;
    
    while (1){
        memset(&info_buffer, NULL, SOCK_SEND_BSIZE);
        temp_nodo = list_search(connections, handler_id);
        n = read(temp_nodo->newsockfd, info_buffer, SOCK_SEND_BSIZE);
        if (n != 0){
            if (n < 0){
                perror("Error on socket reading");
                exit(-1);
            }
            
            printf("[MESSAGE RECEIVED]\n");
            // Check for fall detection
            if (info_buffer[0])
                printf("FALL DETECTED\n");
            
            // Check for location information
            if (info_buffer[1]){
                buffer_i = 3;
                printf("(latitude, longitude) = (");
                for(char_i = 0; char_i < 17; char_i++, buffer_i++)
                    latitude_string[i] = info_buffer[buffer_i]; 
                for(char_i = 0; char_i < 17; char_i++, buffer_i++)
                    longitude_string[i] = info_buffer[buffer_i]; 
                printf("%s,%s)\n", latitude_string, longitude_string);
            }
            
            // Check for data window
            if (info_buffer[2]){                    // If data window is present
                printf("Received data: ");
                for(buffer_i = 37; buffer_i < SOCK_SEND_BSIZE; buffer_i++){
                    putchar(info_buffer[buffer_i]);
                    if ((buffer_i - 37) % 10 == 0)              // put a space after each number
                        putchar(' ');
                }
            }
        }
        else{
            printf("Client %d disconected, closing connection and removing unused nodo\n", handler_id);
            close(temp_nodo->newsockfd);
            list_remove(nodo_list, handler_id);
            pthread_exit(NULL);
        }
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