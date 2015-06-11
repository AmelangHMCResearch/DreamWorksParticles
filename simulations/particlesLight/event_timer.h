/*
 * event_timer.h
 * Author: Savannah Baron
 * Holds interface for doing event timing in CUDA
 */

#ifndef __EVENTTIMER_H__
#define __EVENTTIMER_H__

#include <cuda.h>
#include <cuda_runtime.h> 
#include "ctime"
#include <sys/time.h>

class EventTimer
{
    public:
    	EventTimer();

        // Create an event timer which can time
    	// up to numTimers different things
        EventTimer(unsigned int numTimers);

	    ~EventTimer();
	
	    // Starts the timer indicated by timerNum
	    void startTimer(unsigned int timerNum, bool runKernel);

	    // Stops the timer indicated by timerNum
	    void stopTimer(unsigned int timerNum, bool runKernel);

	    // Gets current time of timer indicated by timerNum
	    // Note: Should only be used when timer is stoped
	    float getTime(unsigned int timerNum);

	    // Returns array of timings for all timers
	    float* getTimes();
    private:
        float* m_times;
        struct timeval m_start, m_stop;
        cudaEvent_t m_devstart, m_devstop;
};

#endif
