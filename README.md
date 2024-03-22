# imageAttackDetectionAndDenoising

A diffusion-based denoising approach to mitigate online adversarial image attacks and an FFT-based detector  

# How to use this code

**This code has been tested on Ubuntu 18.04 LTS and ROS melodic with Python 3.6.9 and Python 3.10.0 and GCC 8.4.0.**

Open up 3 terminals

Create a catkin workspace and download the repositories as follows
```
# in terminal 1

mkdir -p ~/defender_ws/src && cd ~/defender_ws/src

git clone https://github.com/r-bahrami/iros_image_attack.git
git clone https://github.com/gaudelbijay/attack-defender.git
git clone https://github.com/ros-perception/vision_opencv -b melodic

cd ..
```

 The attack-defender model *requires Python 3.10.0*. As such, set up a virtual environment for installing the requirements in `requirements.txt`.

```
# in terminal 2

python3.10 -m venv diffusion_env
source diffusion_env/bin/activate
cd <path to defender>/src/attack-defender
pip install -r requirements.txt

cd src/diffusion_model/ && mkdir results
```
Download the trained parameters of diffusion-based denoiser, named `model-100.pt` from this [link](https://stevens0-my.sharepoint.com/:u:/g/personal/mbahrami_stevens_edu/ERYqWUevbeRKoVV6buwxs7ABo4PeSoxzI3Bdlik3iyE-uA?e=bOYAsr) and copy into the `results` folder.

### Build and train the RL-based attacker model
```
# in terminal 1
# cd ~/defender_ws/

catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

source  devel/setup.bash
```

Set up the AirSim and simulation environment settings (Environment 2) as described in [iros_image_attack](https://github.com/r-bahrami/iros_image_attack) package.

**For integration of the attack-defender, in `model_yolo.py`, uncomment the [line 71](https://github.com/r-bahrami/iros_image_attack/blob/main/scripts/node_yolo.py#L71C7-L71C101) and comment line 70. This enforces the simulation to use the denoised images, and thus the RL-baed attacker is trained while the attack-defender is in the loop.**

Run the `./Blocks.sh` environment in terminal 3. 

Run the attack-defender,
```
# in terminal 2
# source diffusion_env/bin/activate

cd ~/defender_ws/src/attack-defender/src/diffusion_model

python node_denoising.py
```

Then train the attacker using

```
# in terminal 1
roslaunch iros_image_attack train.launch
```

After 10 episodes, the code will stop and the trained model will be saved. Afterward, different experiments can be run as described below.

### runtime
```
# in terminal 2
# source diffusion_env/bin/activate
# cd ~/defender_ws/src/attack-defender/src/diffusion_model

python node_denoising.py
```

```
# in terminal 1
roslaunch iros_image_attack run.launch
```


# TO-DO

- [ ] compare the code with this [commit](https://github.com/gaudelbijay/attack-defender/commit/c4eb10ebbdf936bd645dca971ac7de5dde8283d0) to figure out the encoding bug.

#### Integration with attack models.
- [x]  The reinforcement-based attacker of the [RAL-paper](https://doi.org/10.1109/LRA.2023.3280813) based on the forked [iros_image_attack](https://github.com/r-bahrami/iros_image_attack) repository.
      
**To integrate attack-defender model with the RL-based attacker in [iros_image_attack](https://github.com/r-bahrami/iros_image_attack), we only added [one line](https://github.com/r-bahrami/iros_image_attack/blob/main/scripts/node_yolo.py#L71C7-L71C101) in the `node_yolo.py` module to subscribe to the denoised images. Also, we [enforced](https://github.com/r-bahrami/iros_image_attack/commit/f1597849c24660a5bc96c8085a8234925ad00806#diff-6944d0d7a9300e01c1c450fec04fd9a0bf5a5504d0667d9ced074175e38533f5R144) the image encoding `encoding="bgr8"` for the published attacked images in the `node_image_attacker.py` module.**

For integration to take place, in `model_yolo.py`, uncomment the [line 71](https://github.com/r-bahrami/iros_image_attack/blob/main/scripts/node_yolo.py#L71C7-L71C101) and comment line 70. This enables the simulation to run using the denoised images.

- [x] integrate the FFT code for online attack detection. 
- [ ] update the FFT code for online attack detection in 2D space of Height x Width where we have a better parameterized threshold for detection. 
- [ ] add the training details 


## Acknowledgement

We used [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo for the implementation of the training of our diffusion model of the attack-defender. 

We used [iros_image_attack](https://github.com/r-bahrami/iros_image_attack) repo for the reinforcement-based attacker. 
