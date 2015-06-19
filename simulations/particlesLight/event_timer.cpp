#include "event_timer.h"
#include "ctime"
#include "stdio.h"
#include <sys/time.h>


EventTimer::EventTimer() 
{
    _times = NULL;
    _start = NULL;
    _stop = NULL;
    _devStart = NULL;
    _devStop = NULL;
}

EventTimer::EventTimer(unsigned int numTimers)
{
    _times = new float[numTimers];
    _start = new struct timeval[numTimers];
    _stop = new struct timeval[numTimers];
    _devStart = new cudaEvent_t[numTimers];
    _devStop = new cudaEvent_t[numTimers];

    for (int i = 0; i < numTimers; ++i) {
        _times[i] = 0;
    }
}

EventTimer::~EventTimer()
{  
    delete [] _times;
    delete [] _start;
    delete [] _stop;
    delete [] _devStart;
    delete [] _devStop;
}

void EventTimer::startTimer(unsigned int timerNum, bool runKernel)
{ 
    if (runKernel) {
        cudaEventCreate(&_devStart[timerNum]);
        cudaEventCreate(&_devStop[timerNum]);
        cudaEventRecord(_devStart[timerNum]);
    }
    else {
        gettimeofday(&_start[timerNum], 0);
    }
}

void EventTimer::stopTimer(unsigned int timerNum, bool runKernel)
{
    float tempTime = 0;
    if (runKernel) {
        cudaEventRecord(_devStop[timerNum]);
        cudaEventSynchronize(_devStop[timerNum]);
        cudaEventElapsedTime(&tempTime, _devStart[timerNum], _devStop[timerNum]);
    }
    else {
        gettimeofday(&_stop[timerNum], 0);
        tempTime = (float)(1000.0 * (_stop[timerNum].tv_sec - _start[timerNum].tv_sec)
                   + (0.001 * (_stop[timerNum].tv_usec - _start[timerNum].tv_usec)));
    }
    _times[timerNum] += tempTime;
}

float EventTimer::getTime(unsigned int timerNum)
{
    return _times[timerNum];
}

float* EventTimer::getTimes()
{
    return _times;
}