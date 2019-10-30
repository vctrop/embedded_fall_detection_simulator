// This is a small utility lib to enable periodic signals
// Third party implementation (unknown)
	
/*
 * Each periodic thread is allocated a signal between SIGRTMIN to SIGRTMAX: we
 * assume that there are no other uses for these signals.
 * All RT signals must be blocked in all threads before calling make_periodic()
 */

#ifndef _PERIODIC_SIGNALS_H
#define _PERIODIC_SIGNALS_H

#include <signal.h>
#include <time.h>

struct periodic_info{
	int sig;
	sigset_t alarm_sig;
};

int make_periodic (int unsigned us_period, struct periodic_info *info);
void wait_period (struct periodic_info *info);

#endif