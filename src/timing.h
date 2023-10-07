/*
 * timing
 *
 *  Created on: Oct 25, 2011
 *  Author      : Joao Carlos jcfgonc@gmail.com
 *  License     : MIT License
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <time.h>

#ifndef _WIN32
#include <sys/time.h>
#endif

/**
 * returns the time difference between this check and last (clock precision)
 */
double clockDelta() {
	static clock_t oldclock = 0;
	clock_t newclock = clock();
	double diff = (double) (newclock - oldclock);
	oldclock = newclock;
	return diff / (double) CLOCKS_PER_SEC;
}

double clockTicks() {
	clock_t c = clock();
	return (double) c / (double) CLOCKS_PER_SEC;
}

/**
 * returns the CPU's current Time Stamp Counter (TSC)
 */
__inline__ long long int rdtsc() {
	uint lo, hi;
	__asm__ __volatile__ ( // serialize
			"xorl %%eax,%%eax \n        cpuid"
			::: "%rax", "%rbx", "%rcx", "%rdx");
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return (long long int) hi << 32 | lo;
}

//function retrieved from nvidia's SDK

// Helper function to return precision delta time for 3 counters since last call based upon host high performance counter
// *********************************************************************
double shrDeltaT(int iCounterID) {
	// local var for computation of microseconds since last call
	double DeltaT = -1.0;

#ifdef _WIN32 // Windows version of precision host timer
	// Variables that need to retain state between calls
	static LARGE_INTEGER liOldCount0 = { { 0, 0 } };
	static LARGE_INTEGER liOldCount1 = { { 0, 0 } };
	static LARGE_INTEGER liOldCount2 = { { 0, 0 } };

	// locals for new count, new freq and new time delta
	LARGE_INTEGER liNewCount, liFreq;
	if (QueryPerformanceFrequency(&liFreq)) {
		// Get new counter reading
		QueryPerformanceCounter(&liNewCount);

		// Update the requested timer
		switch (iCounterID) {
		case 0: {
			// Calculate time difference for timer 0.  (zero when called the first time)
			DeltaT = liOldCount0.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount0.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount0 = liNewCount;

			break;
		}
		case 1: {
			// Calculate time difference for timer 1.  (zero when called the first time)
			DeltaT = liOldCount1.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount1.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount1 = liNewCount;

			break;
		}
		case 2: {
			// Calculate time difference for timer 2.  (zero when called the first time)
			DeltaT = liOldCount2.LowPart ? (((double) liNewCount.QuadPart - (double) liOldCount2.QuadPart) / (double) liFreq.QuadPart) : 0.0;

			// Reset old count to new
			liOldCount2 = liNewCount;

			break;
		}
		default: {
			// Requested counter ID out of range
			return -9999.0;
		}
		}

		// Returns time difference in seconds sunce the last call
		return DeltaT;
	} else {
		// No high resolution performance counter
		return -9999.0;
	}
#else // Linux version of precision host timer. See http://www.informit.com/articles/article.aspx?p=23618&seqNum=8
	static struct timeval _NewTime; // new wall clock time (struct representation in seconds and microseconds)
	static struct timeval _OldTime0;// old wall clock time 0(struct representation in seconds and microseconds)
	static struct timeval _OldTime1;// old wall clock time 1(struct representation in seconds and microseconds)
	static struct timeval _OldTime2;// old wall clock time 2(struct representation in seconds and microseconds)

	// Get new counter reading
	gettimeofday(&_NewTime, NULL);

	switch (iCounterID)
	{
		case 0:
		{
			// Calculate time difference for timer 0.  (zero when called the first time)
			DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime0.tv_sec + 1.0e-6 * (double)_OldTime0.tv_usec);

			// Reset old time 0 to new
			_OldTime0.tv_sec = _NewTime.tv_sec;
			_OldTime0.tv_usec = _NewTime.tv_usec;

			break;
		}
		case 1:
		{
			// Calculate time difference for timer 1.  (zero when called the first time)
			DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime1.tv_sec + 1.0e-6 * (double)_OldTime1.tv_usec);

			// Reset old time 1 to new
			_OldTime1.tv_sec = _NewTime.tv_sec;
			_OldTime1.tv_usec = _NewTime.tv_usec;

			break;
		}
		case 2:
		{
			// Calculate time difference for timer 2.  (zero when called the first time)
			DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime2.tv_sec + 1.0e-6 * (double)_OldTime2.tv_usec);

			// Reset old time 2 to new
			_OldTime2.tv_sec = _NewTime.tv_sec;
			_OldTime2.tv_usec = _NewTime.tv_usec;

			break;
		}
		default:
		{
			// Requested counter ID out of range
			return -9999.0;
		}
	}

	// Returns time difference in seconds since the last call
	return DeltaT;
#endif
}

#endif /* TIMING_H_ */
