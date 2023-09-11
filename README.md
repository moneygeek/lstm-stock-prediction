# lstm-stock-prediction
Sample code for using LSTMs to predict stock price movements

Created by Jin Choi, PhD [https://www.eddywealth.com/about/jin-choi/](https://www.eddywealth.com/about/jin-choi/)

Installation
------------

Install dependencies using ``pip``::
   
    pip install -r requirements.txt

If you have CUDA installed, you may want to install [pytorch](https://pytorch.org/) separately. Doing so will
significantly speed up model training.

Running
------------

The following will automatically download the data, train the model, and generate charts comparing forecasts
to actual.

    python main.py


Disclaimer
------------

The code in this repository is intended for educational purposes only. Neither Jin nor the companies he works for
is responsible for any losses arising from the use of this code.
