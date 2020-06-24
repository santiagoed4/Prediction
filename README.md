# Virtual detector with machine learning (Well 58-32)

Drilling is a complex operation with uncontrolled parameters and disturbances that could generate non-productive time during the bit trajectory. A possible scenario to improve the operation is proposed in this research, coupling predictions of parameters with anomaly detection algorithms to minimize the cost of the drilling operation.  Machine learning algorithms have contributed to both approaches uncovering relations between input parameters and the quantity of interest. 

(1) Prediction of rate of penetration for Well 58-32 Mildford (Utah-USA). 
We developed experiments based on static data set analysis. For this reason, the analysis of predictions is developed splitting all the data randomly between training (70%) and test (30%). The data is trained in the model selected (support vector regression, random forest, xgboost and neural networks). The data set for support vector regression and neural networks is standardized to zero mean and variance equal to one to remove effects of differing feature magnitudes.
(2) Anomaly detection predicting part of the interval. 
In the assessment, we have defined a interval of analysis between 3595 ft to 4097 ft in Well 58-32 where 70\% represents training and 30\% testing. The proportion of training and test were defined in a sequence. To detect the anomalies, we need to define a threshold which represent the proportion of outliers (1.5\%). However, this reference requires the analysis of several data set to define a common threshold. For this study, the limit has been arbitrarily defined. Further research, with more data available, will contribute to define a realistic limit.

## Data

https://gdr.openei.org/submissions/1101

### Assistance and design tools

Google Colaboratory allows to develop the coding task of machine learning models. Consequently, the following packages will be used: Scikit-Learn for machine learning algorithms, Pandas for data extraction and preparation and, Matplotlib and Seaborn for data visualization as well as Bokeh for interactive visualization, and Keras and Tensorflow for deep learning algorithms. 

## Built With

* [GoogleColab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) 

## Authors

* **Santiago Echeverri** - 
* **Zaid Al-Ars** Delft University of Technology (The Netherlands)
* **Jayantha P. Liyanage** University of Stavanger (Norway)

## License

This project is developed in the research group Quantum and Computer Engineering from Delft University of Technology (The Netherlands)

## Acknowledgments

Inspiration:
* Unsupervised Anomaly Detection: https://www.kaggle.com/victorambonati/unsupervised-anomaly-detection
* Multivariate Time Series Forecasting with LSTMs in Keras: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
* How To Backtest Machine Learning Models for Time Series Forecasting:  https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
* Data Visualization with Bokeh in Python, Part I: Getting Started: https://towardsdatascience.com/data-visualization-with-bokeh-in-python-part-one-getting-started-a11655a467d4
* How to Develop Machine Learning Models for Multivariate Multi-Step Air Pollution Time Series Forecasting: https://machinelearningmastery.com/how-to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/}

