<img src="peaktk_icon.png" alt="drawing" width="480"/>

# PEAKTK: An Open Source Toolkit for Peak Forecasting in Energy Systems

PEAKTK is a toolkit designed to help researchers, utility companies, and organizations predict the peak day of the month and peak hours of the day of the grid/micro grid. The toolkit is written in Python 3.

# Why PEAKTK?

 Quote from the paper [PEAKTK paper]() explaining the goal of the toolkit:

  > Our goal is to improve reproducibility of energy systems research by providing a common framework for evaluating and comparing new peak forecasting algorithms. 
  > Further, PeakTK provides libraries to enable researchers and practitioners to easily incorporate peak forecasting methods into their research when implementing higher level grid optimizations.

# PeakTK's Features

In this toolkit, we provide reference implementations of a range of peak forecasting techniques as follows:

* Peak day of the year prediction algorithms
	* CPP Approach
	* Stopping Approach
	* Probabilistic Approach
* Peak day of the month prediction algorithms 
	* Extreme temperature
	* VPeak
	* Probabilistic Approach (monthly)
* Peak hour of the day prediction algorithms
	* LSTM-based hourly probabilistic classification 
	* LSTM-based hourly demand prediction
* Peak demand prediction
	* LSTM-based demand forecasting
	* SARIMA
	* SVR-based demand forecasting

PeakTK also includes reference energy datasets for for experimentation, quantitiave comparisons as follows:

* ISO New England dataset  
* ESO dataset
* Smart* Apartment dataset

All datasets come with weather dataset

# Installation

We will provide `pip install` after published.












