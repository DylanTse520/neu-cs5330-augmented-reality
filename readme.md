# Project 4: Calibration and Augmented Reality
This is Project 3: Real-time 2D Object Recognition by Yixiang Xie for CS5330 Pattern Recognition & Computer Vision. Demo link: https://drive.google.com/file/d/11xr3kna2roguuYvSC6OMji48qHacTay3/view?usp=share_link

The code is written in C++ with OpenCV4 on Visual Studio Code on macOS Sonoma 14.0 and compiled with a CMakeLists.txt file.

# Run the code
In order to run the code, use command line to run the bash script by ```./run.sh```. To try different configurations, change ```./run.sh``` in the format of ```./2DObjRecog <--km> <--knn=[value]> ```. The ```--km``` is optional to enable adaptive thresholding, which uses k-means clustering with k=2 to cluster the image pixels into two groups, and use the mean value of these two centroids as the thresholding value. The ```--knn=[value]``` is optional to enable k-nearest neighbors classifier with k=[value].

# Extensions
For extensions, I developed a GUI to help labeling objects and collect training data. Press ```n``` during the system running to enable this. You should input the object number and the object label to add a new label. I have 17 objects in 13 categories in total as training objects. The data can be seen in ```./resources/data.csv```. The system allows you to classify multiple object in one time. Just put the objects in the frame to see it. It also allows for showing unknown objects. Unknown objects will be labelled as "Unknown". I also developed k-means adaptive thresholding. Use the optional parameter ```--km``` to enable this.

# Travel days used
2 travel days