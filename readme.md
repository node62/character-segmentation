# Charecter Segmentation 

Character/ image segmentation code to separate characters inside a circle from the sample images. This doesn't require the use of training data.\
sample input:\
![1](https://github.com/node62/character-segmentation/assets/111416348/e243c5b2-e480-4c9a-a39e-8aa7c8e559b7)

sample output:\
![result1](https://github.com/node62/character-segmentation/assets/111416348/d271ac3f-dafb-4fac-8655-868b5eab8947)

The code provided collects all the sample images and stitches them together for result.

A minimum satuaration of 128 set first, then contours are made to detect the circular region and everything except that region is masked out. Then Kmeans segmentation algorithm is run over this region, which is basically assigning same colors to region that are close enough and have same pixel value. Assuming that the background circle has more area than the inside character, then number of pixel of each color is counted and everything except the least appearing values are masked out. Which is our output image extracted from the circular region.
