# imageAttackDetectionAndDenoising

# TO-DO

1. **Pipeline to feed/feedback diffusion model (August 6)** (done)

Subscribe ``/attack_generator_node/attacked_image`` ros node and feed the images from this node to diffusion model. 
After this, publish denoised image from ``diffusion_model/denoised_image`` node, this will publish denoised image continously. In ``model_yolo.py`` replace sub_attacked_image (line70) to subs_denoised_image by subscribing ``/diffusion_model/denoised_image``

2. **Updating diffusion model and Attack detection code (August 6)** (done)
Check code (diffusion_model)

3. Two ROS nodes running from different Python versions (August 6) (DONE)

4. Downgrading the diffusion model for Python 3.6 (August 10)

5. Write an email to Hyung-Jin Yoon for the updated code (August 7 (a little bit flexible)) -- discussed with Hamid on Friday. 

6. Incorporating diffusion model with current simulation (August 12)

7. Implementing different Attack models (August 30)

8. Multiple Experiment and Improvement (September 8)

9. Paper Writing (September 14)

=======================
- Following the replicability and reproducibility guidelines of [Responsible Conduct of Research (RCR)](https://about.citiprogram.org/series/responsible-conduct-of-research-rcr/) outlined for the federally funded research projects:

- make this one into the form of a ros package
- add dependencies
- add the training details
- use [rosbag](http://wiki.ros.org/rosbag) to record every experiment that will be used in the paper. Experiments should be named and saved properly so there will be a one-to-one correspondence between the files saved and the results presented in the paper. A readme file can be used to explain the data structure of experiments.
- raw data and large files can be saved on OneDrive and referred to here.
- not every file/code/data mentioned above will be published for the public, but they should be saved for internal references and further follow-ups.
