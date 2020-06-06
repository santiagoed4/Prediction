# Evaluation of machine learning for optimization and anomaly detection in offshore drilling operations

Studies of drilling operation have been focused on control parameter optimization to improve the rate of penetration and mechanical specific energy. Drilling is a complex operation with uncontrolled parameters and disturbances that could generate non-productive time during the bit trajectory. The operation is not fully automated increasing the likelihood of misbehavior events. A possible scenario to improve the operation is proposed in this research, coupling predictions of parameters with anomaly detection algorithms to minimize the cost of the drilling operation. It means that at the same time that we are optimizing we need to issue an alert in case of misbehavior. Machine learning algorithms have contributed to both approaches uncovering relations between input parameters and the quantity of interest. This research is conducted based on the following structure: First, machine learning models have been implemented with incremental training data available to predict the rate of penetration. Second, detection of misbehavior models of control, uncontrolled and response parameters have been integrated into the algorithm. Our experiments showed that random forest is a competent machine learning algorithm to predict the rate of penetration with a performance error (root mean squared error) of 2,92 m/hr (9,57 ft/hr) in static analysis and 4,43 m/hr (14,55 ft/hr) average error increasing the availability of data. Furthermore, isolation forest represents a flexible method detecting anomalies in the context of unsupervised learning. Both methods, random forest and isolation forest, performance under a similar structure with incremental data architecture. Algorithms for anomaly detection exposed between 52 and 69 anomalies over 6511 points. Results indicated that just one method could miss the detection of a critical event. Finally a virtual detector is proposed with an architecture of five layers to optimize drilling operations.

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

