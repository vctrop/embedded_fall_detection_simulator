#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

typedef struct node {
    int id;
    int newsockfd;
    struct node* next;
} Node;

Node* node_list;


Node* list_create(void);
Node* list_insert(Node* node, int id, int my_sockfd);
Node* list_search(Node* node, int id);
Node* list_remove(Node* node, int id);


int main(int argc, char *argv[]) {

    return 0; 
}

Node* list_create(void){
    return NULL;
}

Node* list_insert(Node* node, int id, int my_sockfd){
    Node *new = (Node*)malloc(sizeof(Node));

    new -> id = id;
    new -> newsockfd = my_sockfd;
    new -> next = node;
    return new;
}

Node* list_search(Node* node, int id){
    Node *p;

    for(p=node;p!=NULL;p=p->next){
        if(p->id == id) return p;
    }
    return NULL;
}

Node* list_remove(Node* node, int id){
    Node *prev = NULL, *p = node;

    while((p!=NULL) && (p->id != id)){
        prev = p;
        p = p->next;
    }
    if(p == NULL)
        return node;
    if(prev == NULL)
        node = p -> next;
    else{
        prev -> next = p -> next;
    }

    free(p);
    return node;
}
