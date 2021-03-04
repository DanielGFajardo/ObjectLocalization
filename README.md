# ObjectLocalization
Code for detecting the pin for the case study

The code consist in the identification of a pin and transform its position to the robot frame so it can be gripped

# There are three main file

- main.py for generating the artificial dataset 
- CVPinTracking.py which uses openCV to identify the pin
- positionApproximation.py which generates a linear regression model for transforming the pixel coordinates to real world coordinates in the frame of the robot

# Model to be integrated in the enviroment

The model is saved in the .h5 file and can be loaded using the positionApproxiamtionRun.py
