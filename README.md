# Climate Model Downscaling using Deep Learning

Our experiments currently work on Linux and Mac (CPU only).
The experiments works on Python 3.10.9. Other versions should work too.

To run our experiments, please follow the following steps:
  1) You can create a virtual environment first (optional) with python3 -m venv venv and activate it using source venv/bin/activate
  2) install all dependencies using: pip3 install -r requirements.txt
  3) if the data is not installed, run the download.py program (IMPORTANT to change the path in the file to one where you want to store it)
     NOTE that for downscaling we need 2 datasets (low + high resolution). This is approximately 10Gb. 
  4) To run our experiments, it is important to make some changes in the state.json file where the hyperparameters are defined. 
  5) Change the path of the data, make sure the in_channels is the right value (with landcoveryou should +1)
  6) After that, you only need to run the following : python3 downscaling.py -p state.json
     Important! We used wandb (weights and biases) in our experiments. If you want to use wandb, you can uncomment one line of the code (line 303) and put      the projet and name of your run
