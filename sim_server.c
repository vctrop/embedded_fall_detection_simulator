/*#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

#include "linked_list.h"

#define SERVER_PORT 49000
#define MAX_DEVICES 50

pthread_mutex_t mutex_accept = PTHREAD_MUTEX_INITIALIZER;

Node* connections = list_create();
int head_id = 0;

// A thread is assigned to each connected device
void* device_handler(void* id);

int main(int argc, char *argv[]) {
    int serv_sockfd, new_sockfd, port_num, temp_ret;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t cli_addr_len;
    pthread_t thread;
    
    // Unbound server socket with protocol and type definition 
    serv_sockfd = socket(AF_INET, SOCK_STREAM, NULL);
    if (serv_sockfd < 0){
        perror("Socket creation error");
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
        perror("Bind error");
        exit(-1);
    }
    
    // Listen for socket connections
    listen(serv_sockfd, MAX_DEVICES);
    cli_addr_len = sizeof cli_addr;
    while (1){
        // Extract connection from pending queue and create a new socket of the same type as serv_sockfd
        new_sockfd = accept(serv_sockfd, (struct sockaddr*) &cli_addr, &cli_addr_len);
        if (new_sockfd < 0){
            perror("Accept error");
            exit(-1);
        }
        
        // Insert a node in the connections list and create a thread to handle the device connection
        pthread_mutex_lock(&mutex_accept);
            connections = list_insert(connections, head_id, new_sockfd);
            pthread_create(&thread, NULL, device_handler, (void*) head_id);
            head_id++;
        pthread_mutex_unlock(&mutex_accept);
    }   
    
    return 0; 
}

void* device_handler(void* id){
    int dev_id = (int) id;
    
    
    
}

