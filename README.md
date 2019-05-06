# Face and Eye detection using OpenCV 
## GOAL
Covering basics of face detection using Haar Feature-based Cascade Classifiers
And extended the same for eye detection etc.

### Description
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by [Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

A Haar-Feature is just like a kernel in CNN, except that in a CNN, the values of the kernel are determined by training, while a Haar-Feature is manually determined.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

![alt text][haar]

[haar]: ./imgs/haar_features_1.jpg
Numerically, they might look something like this:
![alt text][haarnum]


[haarnum]: ./imgs/haar_num.png

As explained here, each the 3x3 kernel moves across the image and does matrix multiplication with every 3x3 part of the image, emphasizing some features and smoothing others.

Haar-Features are good at detecting edges and lines. This makes it especial effective in face detection. 

However, because Haar Features have to be determined manually, there is a certain limit to the types of things it can detect. If you give classifier (a network, or any algorithm that detects faces) edge and line features, then it will only be able to detect objects with clear edges and lines. Even as a face detector, if we manipulate the face a bit (say, cover up the eyes with sunglasses, or tilt the head to a side), a Haar-based classifier may not be able to recognize the face. A convolutional kernel, on the other hand, has a higher degree of freedom (since it’s determined by training), and could be able to recognize partially covered faces (depending on the quality of the training data).

On the plus side, because we don’t need to train Haar-Features, we can create a classifier with a relatively small dataset. All we have to do is train the weightings for each feature (i.e. which Haar-feature should be used more?) which allows us to train the classifier well without a lot of training images. In addition, it also has a higher execution speed, as Haar-based classifiers typically involve less computations.

What triggered this small investigation into Haar-based classifiers is this model that recognizes emotions. Last year, at a fair, I came across a emotion recognition system. However, it didn’t use neural networks. I was curious if I could find a emotion recognition algorithm based completely on CNNs.

Taking a brief look into this model, I saw that it used OpenCV’s Haar-based cascade classifier to detect faces. After finding faces, the team then trained their own CNN to recognize the emotion on the face.

Because it used a Haar-based classifier, I couldn’t really call it an algorithm based completely on convolutional neural networks. What if I switched out the Haar-based classifier for the MTCNN face recognition system?

Originally, it loaded a Haar-based classifier. I switched it out for an MTCNN detector:
(Check this link)[https://towardsdatascience.com/whats-the-difference-between-haar-feature-classifiers-and-convolutional-neural-networks-ce6828343aeb]

Now, all possible sizes and locations of each kernel are used to calculate lots of features. (Just imagine how much computation it needs? Even a 24x24 window results over 160000 features). For each feature calculation, we need to find the sum of the pixels under white and black rectangles. To solve this, they introduced the integral image. However large your image, it reduces the calculations for a given pixel to an operation involving just four pixels. Nice, isn't it? It makes things super-fast.

But among all these features we calculated, most of them are irrelevant. For example, consider the image below. The top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applied to cheeks or any other place is irrelevant. So how do we select the best features out of 160000+ features? It is achieved by Adaboost.

![alt text][haar2]


[haar2]: ./imgs/haar_2.png



For this, we apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. Obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that most accurately classify the face and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then the same process is done. New error rates are calculated. Also new weights. The process is continued until the required accuracy or error rate is achieved or the required number of features are found).

The final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can't classify the image, but together with others forms a strong classifier. The paper says even 200 features provide detection with 95% accuracy. Their final setup had around 6000 features. (Imagine a reduction from 160000+ features to 6000 features. That is a big gain).

So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or not. Wow.. Isn't it a little inefficient and time consuming? Yes, it is. The authors have a good solution for that.

In an image, most of the image is non-face region. So it is a better idea to have a simple method to check if a window is not a face region. If it is not, discard it in a single shot, and don't process it again. Instead, focus on regions where there can be a face. This way, we spend more time checking possible face regions.

For this they introduced the concept of Cascade of Classifiers. Instead of applying all 6000 features on a window, the features are grouped into different stages of classifiers and applied one-by-one. (Normally the first few stages will contain very many fewer features). If a window fails the first stage, discard it. We don't consider the remaining features on it. If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region. How is that plan!

The authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in the first five stages. (The two features in the above image are actually obtained as the best two features from Adaboost). According to the authors, on average 10 features out of 6000+ are evaluated per sub-window.

So this is a simple intuitive explanation of how Viola-Jones face detection works. Read the paper for more details or check out the references in the Additional Resources section.

## My output

![alt text][out]


[out]: ./imgs/out.png

## Results
Was able to detect face and eyes.

### Programing Language Used: 
- Python
### Tools used : 
- Numpy
- Pandas
- Matplotlib
- OpenCV2 

### Steps to run my project

- Download or clone this project
- Check respective documentations for installing [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html), [NumPy](https://docs.scipy.org/doc/numpy/user/install.html), [OpenCV2](https://pypi.org/project/opencv-python/) using pycharm
- All you have to do is run this [Face_and_Eye_Detection.py ](./Face_and_Eye_Detection.py) file. 


## Rerences :

- [Udemy](https://www.udemy.com/master-computer-vision-with-opencv-in-python/learn/v4/t/lecture/5860732?start=0)
- [ Viola and Jones Paper on Face Detection ](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) 
