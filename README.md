CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ziwei Zong
* Tested on: Windows 10, i7-5500 @ 2.40GHz 8GB, GTX 950M (Personal)

Descriptions
--------------------------
Part 1 : CPU Scan & Compaction (see file cpu.cu)
Part 2 : Naive Scan (naive.cu)
Part 3 : Work-Efficient Scan & Compaction (efficient.cu)
Part 4 : Thrust Scan (thrust.cu)
Part 5 : Radix Sort (in file thrust.cu, RadixSort::sort)

Block Sizes Optimization
--------------------------
|            |  32 |  64 | 128 | 256 | 512 |1024|
| block_naive|0.062|0.061|0.060|0.062|0.064|0.078|
|   block_eff|0.139|0.139|0.140|0.142|0.148|0.155|
|block_thrust|1.060|1.180|1.200|1.100|1.029|1.090| 
(ms)
Thus, I choose block size 128 for naive scan, 64 for efficient scan and 512 for thrust scan.

Output 
--------------------------

****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
==== cpu scan, power-of-two ====
StreamCompaction::CPU::scan : exclusive prefix sum.
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6229 ]
==== cpu scan, non-power-of-two ====
StreamCompaction::CPU::scan : exclusive prefix sum.
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6146 6190 ]
    passed
==== naive scan, power-of-two ====
         GPU time for naive scan : 0.0696ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6229 ]
    passed
==== naive scan, non-power-of-two ====
         GPU time for naive scan : 0.0676ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6146 6190 ]
    passed
==== work-efficient scan, power-of-two ====
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  26   0 ]
         GPU time for efficient scan : 0.1403ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6229 ]
    passed
==== work-efficient scan, non-power-of-two ====
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  44   8 ]
         GPU time for efficient scan : 0.1403ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6146 6190 ]
    passed
==== thrust scan, power-of-two ====
         GPU time for thrust scan : 128.8397ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6203 6229 ]
    passed
==== thrust scan, non-power-of-two ====
         GPU time for thrust scan : 1.1305ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 6146 6190 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   2 ]
    passed
==== cpu compact with scan ====
StreamCompaction::CPU::scan : exclusive prefix sum.
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   2 ]
    passed

*****************************
**        Radix Sort       **
*****************************
==== Radix Sort, power-of-two ====
    [   4   7   2   6   3   5   1   0 ]
    [   4   2   6   0   7   3   5   1 ]
