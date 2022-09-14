# Descriptions
In this folder, we generate samples in Directory **VBF_H5pp_ww_jjjj**, **VBF_H5mm_ww_jjjj**, and **VBF_H5z_ww_jjjj**. Then the jet samples are selected in **extract.py** and the selected samples are divided into training, valid, and testing dataest in **convert.py**. For the last two step, they are executed in **run.sh**. The datasets are saved in folder **samples_kappa* ** (They are not archived in GitHub since they are too big.), and ready to be used for machine learning.

# Coding script Usage
Here are the fundamental functional script for the data processing. 
* **convert (py)**: convert the selected jet samples into training, valid, and testing datasets.
* **extract (py)**: select the proper jet from the Delphes samples and extract their jet images.
* **plot_jet_sample (py)**: plot the jet sample image in average for all class of data.
* **run (sh)**: run through all the step at once, including extract, plot, and convert.
* **rundelphes (sh)**: Use Delphes to generate detector-level information from Pythia sample.
* **tfr_utils (py)**: helper functions for creating tf.train.feature of different types. 
* **writeTFR (py)**: helper functions that transform the data from npy to tfrecord.


