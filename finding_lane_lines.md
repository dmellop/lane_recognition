
# **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x112a4cda0>




![png](output_4_2.png)


**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(grayscale(image),cmap='gray')  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x115cabbe0>




![png](output_8_2.png)



```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(canny(grayscale(image),100,250),cmap='gray') #50, 150
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x1187727b8>




![png](output_10_2.png)



```python
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(canny(gaussian_blur(grayscale(image),7),100,250),cmap='gray') #5 ,50, 150
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x11b35edd8>




![png](output_12_2.png)



```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with

    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```


```python
#reading in an image
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(region_of_interest(canny(gaussian_blur(grayscale(image),7),100,250),vertices),cmap='gray') #5 ,50, 150
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x11ce3f080>




![png](output_14_2.png)



```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```


```python
#reading in an image
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(
    region_of_interest(
        hough_lines(
            canny(
                gaussian_blur(
                    grayscale(image),
                    7),
                150,250),
        1,np.pi/180,60,40,20),
        vertices)
    ,cmap='gray') #5 ,50, 150
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x128dc5f60>




![png](output_16_2.png)



```python
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```


```python
#reading in an image
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(
    region_of_interest(
        weighted_img(
        hough_lines(
            canny(
                gaussian_blur(
                    grayscale(image),
                    7),
                100,250),
        rho,theta,threshold,min_line_len,max_line_gap),
            image,0.8,1.0,0.0),
        vertices)
    ,cmap='gray') #5 ,50, 150
```

    This image is: <class 'numpy.ndarray'> with dimesions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x129102278>




![png](output_18_2.png)



```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)




# Python 3 has support for cool math symbols.


```


```python
#reading in an image
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(
    region_of_interest(
        weighted_img(
        hough_lines(
            canny(
                gaussian_blur(
                    grayscale(image),
                    7),
                100,250),
        rho,theta,threshold,min_line_len,max_line_gap),
            image,0.8,1.0,0.0),
        vertices)
    ,cmap='gray') #5 ,50, 150
```

## Pipeline

## Test on Images

Now you should build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```

run your solution on all test_images and make copies into the test_images directory).

## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```


```python
class processing(object):
    def __init__(self):
        self.left_x1   = 0
        self.left_x2   = 0
        self.left_y1   = 0
        self.left_y2   = 0
        self.right_x1  = 0
        self.right_x2  = 0
        self.right_y1  = 0
        self.right_y2  = 0
        
        
    def process_image(self,image,g=5,lt=100,ht=250,p1=450,p2=320,p3=490,p4=320,rho=2,theta=np.pi/180,
                      threshold=15,mll=40,mlg=20,α=0.8,β=1.0,λ=0,color=[255, 0, 0],thickness=10):

        def get_slope(x1,y1,x2,y2):
            return ((y2-y1)/(x2-x1))

        edges             = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(g,g),0),lt,ht)    

        imshape           = image.shape
        vertices          = np.array([[(0,imshape[0]),(p1,p2),(p3,p4),
                                       (imshape[1],imshape[0])]], dtype=np.int32)

        mask              = np.zeros_like(edges)   
        ignore_mask_color = 255  

        cv2.fillPoly(mask,vertices,ignore_mask_color)
        masked_edges      = cv2.bitwise_and(edges, mask)
        line_image        = np.copy(image)*0

        lines             = cv2.HoughLinesP(masked_edges,rho,theta, 
                                            threshold,np.array([]),
                                            mll,mlg)
        bottom = image.shape[0]
        top    = int(bottom*0.6) 


        left_x1s = []
        left_y1s = []
        left_x2s = []
        left_y2s = []
        right_x1s = []
        right_y1s = []
        right_x2s = []
        right_y2s = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = get_slope(x1, y1, x2, y2)

                if slope < 0:
                    if slope > -.5 or slope < -.8:
                        continue        
                    left_x1s.append(x1)
                    left_y1s.append(y1)
                    left_x2s.append(x2)
                    left_y2s.append(y2)
                    
                    
                else:
                    if slope < .5 or slope > .8:
                        continue        
                    right_x1s.append(x1)
                    right_y1s.append(y1)
                    right_x2s.append(x2)
                    right_y2s.append(y2)
                    

        try:
            avg_right_x1 = int(np.mean(right_x1s))
            avg_right_y1 = int(np.mean(right_y1s))
            avg_right_x2 = int(np.mean(right_x2s))
            avg_right_y2 = int(np.mean(right_y2s))
            right_slope  = get_slope(avg_right_x1, avg_right_y1, avg_right_x2, avg_right_y2)

            right_y1 = top
            right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
            right_y2 = bottom
            right_x2 = int(avg_right_x2 + (right_y2 - avg_right_y2) / right_slope)
            
            #cv2.line(image, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
            
            self.right_x1 = right_x1
            self.right_x2 = right_x2
            self.right_y1 = right_y1
            self.right_y2 = right_y2
            
            
        except ValueError:
            pass
        
        cv2.line(image, (self.right_x1,self.right_y1), (self.right_x2,self.right_y2), color, thickness)

        try:
            avg_left_x1 = int(np.mean(left_x1s))
            avg_left_y1 = int(np.mean(left_y1s))
            avg_left_x2 = int(np.mean(left_x2s))
            avg_left_y2 = int(np.mean(left_y2s))
            left_slope = get_slope(avg_left_x1, avg_left_y1, avg_left_x2, avg_left_y2)

            left_y1 = top
            left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
            left_y2 = bottom
            left_x2 = int(avg_left_x2 + (left_y2 - avg_left_y2) / left_slope) 
            
            self.left_x1 = left_x1
            self.left_x2 = left_x2
            self.left_y1 = left_y1
            self.left_y2 = left_y2
            
        except ValueError:
            pass
        
        cv2.line(image, (self.left_x1,self.left_y1), (self.left_x2,self.left_y2), color, thickness) 

        color_edges = np.dstack((edges, edges, edges)) 

        return cv2.addWeighted(image,α,line_image,β,λ)
```


```python
processor = processing()

image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(processor.process_image(image,g=5,lt=100,ht=170,p1=450,p2=320,p3=490,p4=320,rho=1,theta=np.pi/180,
                  threshold=15,mll=40,mlg=40,α=0.8,β=1.0,λ=0,color=[255, 0, 0],thickness=10))
```




    <matplotlib.image.AxesImage at 0x1314ea978>




![png](output_28_1.png)


Let's try the one with the solid white lane on the right first ...


```python
processor = processing()
white_output = 'white.mp4'
clip1        = VideoFileClip("solidWhiteRight.mp4")
white_clip   = clip1.fl_image(processor.process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    100%|█████████▉| 221/222 [00:08<00:00, 25.57it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4 
    
    CPU times: user 3.85 s, sys: 1.02 s, total: 4.87 s
    Wall time: 10.3 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




**At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
processor = processing()
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(processor.process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    100%|█████████▉| 681/682 [00:27<00:00, 19.92it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4 
    
    CPU times: user 12.3 s, sys: 3.15 s, total: 15.5 s
    Wall time: 28.4 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>




## Reflections

Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?

Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!


Definetively there is a need to finely tune each step of the pipeline to get the best results. Particularly wasn't able to truly extrapolate the lines in all instances. Also the edges are evident in the stream.

Edge detections are most likely to fail with other 'vertical' lines, detecting them as well.

AS seen in the optional challenge, trees and other objects with sharp edges produce noise and line detection where it is not, other lines may create new lane lines which can be dangerous for the logic of the car.

## Submission

If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
processor = processing()
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(processor.process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video extra.mp4
    [MoviePy] Writing video extra.mp4


    100%|██████████| 251/251 [00:24<00:00, 12.67it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: extra.mp4 
    
    CPU times: user 8.97 s, sys: 2.47 s, total: 11.4 s
    Wall time: 26.1 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="extra.mp4">
</video>



