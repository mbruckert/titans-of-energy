# Getting this to work on your system is pretty complex. I didn't bother to simplify the process because everything is going to have to be changed to incorporate with the UI anyways.


# Step 1 make a virtual environment in conda
# Step 2 install pip
# Step 3 clone the repo
# Step 4 clone the appropriate models into the Models folder. Follow their documentation. For testing purposes don't install F5TTS.
# Step 5 pip install -r requirments.txt

# Quick break

# All of the scripts are divided by their function. The testing pipeline script is the main script for running all the scripts together but as you will see, the output fails due to dependency conflict between the F5 module package and XTTS and zonos. XTTS and Zonos can run together without conflicting though.

# Step 6 import the appropriate inference file from the ModelRunFiles folder into the appropriate models folder. Rename each inference file to inference.py

# Step 7 from here you can choose to run the models directly if you change the reference audio and output paths by running python inference.py

# For running the whole pipeline, you will likely run into a dependency conflict. The program will not be a complete failure though, each stage is broken up into their try and catch phases so all of the audio preprocessing will output correctly even if the models fail to run. Handling the dependency conflict is dependent on what models you choose to include. If you only choose to run 1 model at a time, it should work without any dependency conflict. 

# Step 8 choose the correct settings and run python testingPipeline.py

# One thing that will be need to be done in the future is priming (preloading) the model for when using live. Preloading takes about 30-40 seconds but it only has to be done once. In a live situation this will reduce the runtime to realtime responses. 

