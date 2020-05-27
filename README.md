# inkpoint-ml
Machine learning algorithm for generating ink point illustrations, as if created by an ink pen on paper.

This algorithm is VERY MUCH a prototype. Much of the code is quite dirty and not properly organized/maintained.

Uses Python, and requires the following libraries:
- Numpy
- Pillow
- OpenCV2

# Introduction
Very basic gradient descent style algorithm. Currently learns how to draw individual points from a series of samples, and then deterministically applies generated points to recreate a sample image.

Non-deterministic application of points as well as shading is currently under development. 

App.py is the entry point. Run this file to get the algorithm going.

Modify the following lines to generate an image using the pre-calculated weights:
- Line 20
- Lines 34-36

If you want to learn/generate your own weights via the gradient descent algorithm, perform the following operations:
- Uncomment line 26
- Comment out lines 19, 20, 32-38
