// Library containing basic struct and functions to operate with linked lists of socket connections
// Victor O. Costa

#ifndef _LINKED_LIST_H
#define _LINKED_LIST_H

#include <stdlib.h>

// Linked list basic node struct
typedef struct node {
    int id;
    int newsockfd;
    struct node* next;
} Node;

// Returns null pointer of *Node type
Node* list_create(void);

// Inserts a node in the beginning of a given list, with an id and sockfd
Node* list_insert(Node* node, int id, int my_sockfd);

// Searches list and returns first node with matching id
Node* list_search(Node* node, int id);

// Removes node with a given id from the list
Node* list_remove(Node* node, int id);

#endif