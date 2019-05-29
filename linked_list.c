#include "linked_list.h"
#include <stdlib.h>

Node* list_create(void){
    return NULL;
}

Node* list_insert(Node* node, int id, int my_sockfd){
    /* Inserts a node in the beginning of the list */
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
