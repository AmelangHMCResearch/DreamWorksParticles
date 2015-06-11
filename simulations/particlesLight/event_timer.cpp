#include "event_timer.h"
#include "ctime"
#include <sys/time.h>


EventTimer::EventTimer() 
{
    m_times = NULL;
};

EventTimer::EventTimer(unsigned int numTimers)
{
    m_times = new float[numTimers];

    for (int i = 0; i < numTimers; ++i) {
        m_times[i] = 0;
    }
}

EventTimer::~EventTimer()
{  
    delete[] m_times;
}

void EventTimer::startTimer(unsigned int timerNum, bool runKernel)
{ 
    if (runKernel) {
        cudaEventCreate(&m_devstart);
        cudaEventCreate(&m_devstop);
        cudaEventRecord(m_devstart);
    }
    else {
        gettimeofday(&m_start, 0);
    }
}

void EventTimer::stopTimer(unsigned int timerNum, bool runKernel)
{
    float tempTime = 0;
    if (runKernel) {
        cudaEventRecord(m_devstop);
        cudaEventSynchronize(m_devstop);
        cudaEventElapsedTime(&tempTime, m_devstart, m_devstop);
    }
    else {
        gettimeofday(&m_stop, 0);
        tempTime = (float)(1000.0 * (m_stop.tv_sec - m_start.tv_sec)
                   + (0.001 * (m_stop.tv_usec - m_start.tv_usec)));
    }
    m_times[timerNum] += tempTime;
}

float EventTimer::getTime(unsigned int timerNum)
{
    return m_times[timerNum];
}

float* EventTimer::getTimes()
{
    return m_times;
}
