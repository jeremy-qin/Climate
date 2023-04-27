# Climate Model Downscaling using Deep Learning

Our experiments currently work on Linux and Mac (CPU only).
The experiments run on Python 3.10.9. Later versions ought to work fine as well.

## To replicate our experiments, please follow the following steps:
  ### 1) Create a virtual environment first (optional) with the following shell command:
      python3 -m venv venv 
   ### and activate it using: 
      source venv/bin/activate

  ### 2) Install all dependencies with: 
      pip3 install -r requirements.txt

  ### 3) If the data is not installed, run download.py (IMPORTANT to change the path in the file to one where you want to store it)
     NOTE that for downscaling we need 2 datasets (low + high resolution). This should take approximately 10GB of storage. 
     
  ### 4) To run our experiments, it is essential to modify the dataset path in the state.json file where the hyperparameters are defined. You should make sure it matches the path of the dataset in your system. You're also free to change any of the other hyperparameters to your liking. To run a baseline of our experiments, only need to change the path of root_dir and root_highres_dir.

  ### 5) After changing the path, make sure the in_channels is the right value (with landcover you should increase the value by 1, since you're adding one dimension to the image). If running the baseline, keep it at 1.

  ### 6) After that, you only need to execute the command: 
      python3 downscaling.py -p state.json
   Important! We used wandb (weights and biases) in our experiments. If you want to use wandb, you can uncomment one line of the code (line 303) and enter the projet and name of your trial.
