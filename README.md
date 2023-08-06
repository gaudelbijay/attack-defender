# imageAttackDetectionAndDenoising

# TO-DO

1. **Pipeline to feed/feedback diffusion model (August 6)** (done)

Subscribe ``/attack_generator_node/attacked_image`` ros node and feed the images from this node to diffusion model. 
After this, publish denoised image from ``diffusion_model/denoised_image`` node, this will publish denoised image continously. In ``model_yolo.py`` replace sub_attacked_image (line70) to subs_denoised_image by subscribing ``/diffusion_model/denoised_image``

2. **Updating diffusion model and Attack detection code (August 6)** (done)
Check code (diffusion_model)

3. Two ROS node running from different python version (August 6) (DONE)

4. Downgrading the diffusion model for python 3.6 (August 10)

5. Write email to Hyung-Jin Yoon for updated code (August 7 (little bit flexible))

6. Incorporating diffusion model with current simulation (August 12)

7. Implementing different Attack models (August 30)

8. Multiple Experiment and Improvement (September 8)

9. Paper Writing (September 14)
