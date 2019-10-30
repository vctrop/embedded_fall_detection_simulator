CFLAGS= -Wall -pthread -lrt -O2 -std=gnu99
DEVICE_TARGS= periodic_signals.c  sim_detector.c  
SERVER_TARGS= linked_list.c sim_server.c 

all: $(DEVICE_TARGS) $(SERVER_TARGS)
	gcc $(DEVICE_TARGS) -o device $(CFLAGS)
	gcc $(SERVER_TARGS) -o server $(CFLAGS)


.PHONY: clean
clean:
	$(RM) device server *.o
