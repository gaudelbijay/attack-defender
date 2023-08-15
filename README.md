# imageAttackDetectionAndDenoising

# TO-DO

1. **Pipeline to feed/feedback diffusion model (August 6)** (DONE)

Subscribe ``/attack_generator_node/attacked_image`` ros node and feed the images from this node to diffusion model. 
After this, publish denoised image from ``diffusion_model/denoised_image`` node, this will publish denoised image continously. In ``model_yolo.py`` replace sub_attacked_image (line70) to subs_denoised_image by subscribing ``/diffusion_model/denoised_image``

2. **Updating diffusion model and Attack detection code (August 6)** (DONE)
Check code (diffusion_model)

3. Two ROS nodes running from different Python versions (August 6) (DONE)

4. Downgrading the diffusion model for Python 3.6 (August 10) (Escaped by using 3)

5. Write an email to Hyung-Jin Yoon for the updated code (August 7 (a little bit flexible)) -- discussed with Hamid on Friday. 

6. Incorporating diffusion model with current simulation (August 12) (DONE)

7. Implementing different Attack models (August 30)

8. Multiple Experiment and Improvement (September 8)

9. Paper Writing (September 14)

========================================================================================
- Following the replicability and reproducibility guidelines of [Responsible Conduct of Research (RCR)](https://about.citiprogram.org/series/responsible-conduct-of-research-rcr/) outlined for the federally funded research projects:

- make this one into the form of a ros package
- add dependencies
- add the training details
- use [rosbag](http://wiki.ros.org/rosbag) to record every experiment that will be used in the paper. Experiments should be named and saved properly so there will be a one-to-one correspondence between the files saved and the results presented in the paper. A readme file can be used to explain the data structure of experiments.
- raw data and large files can be saved on OneDrive and referred to here.
- not every file/code/data mentioned above will be published for the public, but they should be saved for internal references and further follow-ups.

========================================================================================
# How to run this code
```
mkdir -p icra_ws/src && cd icra_ws/src

git clone https://github.com/stargaze221/iros_image_attack

git clone https://github.com/gaudelbijay/attack-defender

git clone https://github.com/ros-perception/vision_opencv -b melodic

cd ..

catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

source  devel/setup.bash
# set-up the AirSim and simulation environment settings as described in iros_image_attack package.
# run the ./Blocks.sh environemnt. Then
roslaunch iros_image_attack run.launch

```
Setup environment for the defender model *we need to setup different python (version: 3.10) virtual environment*. Install all the requirements from `requirements.txt`

```

python3.10 -m venv diffusion_env

source diffusion_env/bin/activate


cd <path to icra_ws>/src/attack-defender

pip install -r requirements.txt

mkdir results

# copy model-100.pt> from this [link](https://stevens0-my.sharepoint.com/personal/mbahrami_stevens_edu/_layouts/15/onedrive.aspx?ct=1692041444591&or=OWA%2DNT&cid=ffda05e7%2D3cac%2D4eb2%2D4c48%2D49021369a39d&ga=1&WSL=1&id=%2Fpersonal%2Fmbahrami%5Fstevens%5Fedu%2FDocuments%2FAdv%5Fimage%5Fatk%5Fdetection%2Fresults) into results folder.

python node_denoising.py
```
