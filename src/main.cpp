/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <string>
#include <iostream>

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 10;
    const int NPOT = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

    // Scan tests
	
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
    printArray(SIZE, a, true);
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    //printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    //printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);
	
    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
	//printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
	
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
	//*
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    printArray(SIZE, a, true);
	a[SIZE - 1] = 0;
    int count, expectedCount, expectedNPOT;
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
	printArray(SIZE, a, true);//delete
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
	printArray(SIZE, a, true);//delete
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	printf("\n");
	printf("*****************************\n");
	printf("**        Radix Sort       **\n");
	printf("*****************************\n");

	a[0] = 4;
	a[1] = 7;
	a[2] = 2;
	a[3] = 6;
	a[4] = 3;
	a[5] = 5;
	a[6] = 1;
	a[7] = 0;

	zeroArray(8, c);
	printDesc("Radix Sort, power-of-two");
	printArray(8, a, true);//delete
	StreamCompaction::RadixSort::sort(8, c, a);
	printArray(8, c, true);
	//printCmpResult(SIZE, b, c);*/
	std::cin.get();
}
