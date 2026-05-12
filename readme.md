# hybrid-forecaster-ml

this is a simple project to show how to mix classical stats (linear regression) 
with machine learning (lightgbm) for time series.

## why do this?
gbms are great but they suck at predicting trends that go outside the training range. 
this hybrid fixes that by using a linear model for the 'big picture' trend.

## how to use
1. install things: `pip install -r requirements.txt`
2. run it: `python setup.py install`