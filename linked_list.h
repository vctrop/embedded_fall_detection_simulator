#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct node {
    int id;
    int newsockfd;
    struct node* next;
} Node;

Node* list_create(void);
Node* list_insert(Node* node, int id, int my_sockfd);
Node* list_search(Node* node, int id);
Node* list_remove(Node* node, int id);

#endif