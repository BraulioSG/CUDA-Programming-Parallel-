# Sliding Window Sum
## Colaborators
- Braulio Solorio
- Tijash Salamanca
  
## About
 a CUDA kernel that takes a vector of floats with
a minimum of 5 elements, applies a sliding window of 2 elements backwards and 2 forwards, calculates
the average of those elements and the current element and saves the result in the same index of the
current element but in another vector. The maximum size of the vector is the maximum number of threads
per block in X
