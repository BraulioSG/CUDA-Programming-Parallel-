# CUDA Programming Projects
This repository contains my projects of my university subject Advance Parallel Programming in CUDA
All these projects were developed using Visual Studio Community 2022 with the CUDA project template.
On this repository, only the main files are uploaded `.cu` files.

## Build A project
**Requirements:**
- Visual Studio 2022
- Nvidia CUDA Toolkit

**Steps**
1. Open Visual Studio 2022
2. Create a new CUDA project
3. Copy and Paste the `.cu` files
4. Compile & Run de project
>[!NOTE]
>More specific or additional requirements / steps will be mentioned inside the folder of the project

## Midterm 1: Sliding Window Sum
launch a CUDA kernel that takes a vector of floats with
a minimum of 5 elements, applies a sliding window of 2 elements backwards and 2 forwards, calculates
the average of those elements and the current element and saves the result in the same index of the
current element but in another vector. The maximum size of the vector is the maximum number of threads
per block in X
