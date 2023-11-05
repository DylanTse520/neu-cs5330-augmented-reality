# Project 4: Calibration and Augmented Reality
This is Project 4: Calibration and Augmented Reality by Yixiang Xie for CS5330 Pattern Recognition & Computer Vision.

The code is written in C++ with OpenCV4 on Visual Studio Code on macOS Sonoma 14.0 and compiled with a CMakeLists.txt file.

# Run the code
In order to run the code, use command line to run the bash script by running ```./run.sh```.

By default the script will run the calibration part, which detects a chessboard for its corners and allows for taking calibration images by pressing "s". You need at least 5 images for calibration.

After calibration, to view a virtual object on the chessboard, change line 5 in ```./run.sh``` to ```./ar <static image path containing a chessboard>```. The second parameter is optional. When leave out, the program will detect a chessboard and project a Utah teapot onto it. You can also pass a path to a static image containing a chessboard as the second parameter. The program will insert the teapot to the image and display it.

To view the robust features detection, change line 5 in the script to ```./feature```. By default the program shows SURF features. To change between features, press "u" for SURF features, press "i" for SIFT features, press "h" for Harris corners or press "t" for Shi-Tomasi corners.

Press "q" to quit either program.

# Extensions
For extensions, the program allow for detecting four different robust features. To test them, run the script with ```./feature```, and press "u" for SURF features, press "i" for SIFT features, press "h" for Harris corners or press "t" for Shi-Tomasi corners. I also hid the chessboard underneath a white mask. To test this, run the script with ```./ar``` and put a chessboard in the frame. I also allow using static images with chessboard to demonstrate inserting teapot in it. To test this, run the script with ```./ar <static image path containing a chessboard>``` and specify an image path with a chessboard in it.

# Travel days used
1 travel days