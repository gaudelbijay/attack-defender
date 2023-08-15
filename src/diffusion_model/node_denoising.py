import rospy 
import torch 
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

class Denoise(object):
    def __init__(self, 
                 image_size=448, 
                 dim=64, 
                 timestamp=1000, 
                 sampling_timesteps=250, 
                 flash_attn=False, 
                 objective='pred_noise',
                 train_dir='../data/raw/',
                 test_dir='../data/test/',
                 batch_size = 1,
                 device = "cuda") -> None:
        
        self.image_size = image_size
        
        self.dim = dim 
        self.flash_attn = flash_attn 
        self.timestamp = timestamp
        self.sampling_timesteps = sampling_timesteps
        self.objective = objective 
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.device = device

        self.model_init()
    
    def model_init(self,):
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True
            ).to(self.device)
        
        self.diffusion = GaussianDiffusion(
        model = self.model,
        image_size = self.image_size,
        timesteps = self.timestamp,           # number of steps
        sampling_timesteps = self.sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        objective = self.objective,
        ).to(self.device)

        self.trainer = Trainer(
            diffusion_model=self.diffusion, 
            folder=self.train_dir,
            )
        self.trainer.load(100) # 100 => last saved model

        rospy.init_node("attacked_image_denoiser", anonymous=False)

        self.sub_attacked_img = rospy.Subscriber(
            "/attack_generator_node/attacked_image", Image, queue_size=10, callback = self.call_back,
            )
        
        self.pub_denoised_img = rospy.Publisher(
            "/diffusion_model/denoised_image", Image, queue_size=10,
        )
        self.bridge = CvBridge()

    def denoise(self, image, t=torch.tensor([1])):
        t = t.to(self.device)
        predicted_image = self.diffusion.to(self.device).model_predictions(image, t)[1]
        return predicted_image
    
    def call_back(self, image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8") 

            # publish directly without denoising
            # image_tensor = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            # self.pub_denoised_img.publish(image_tensor)

            # denoise the image
            image_tensor = torch.from_numpy(cv_image.transpose(2,0,1)).float().unsqueeze(0)
            denoised_image_tensor = self.denoise(image_tensor.to(self.device))

            if denoised_image_tensor.is_cuda:
                denoised_image_tensor = denoised_image_tensor.cpu().detach()

            # # convert denoised image back to opencv format 
            denoised_cv_image = denoised_image_tensor.squeeze(0).byte().numpy().transpose(1,2,0)

            # show image in opencv
            # cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
            # cv2.imshow('Display', denoised_cv_image)
            # cv2.waitKey(1)

            #convert image to ros message 
            denoised_ros_image_msg = self.bridge.cv2_to_imgmsg(denoised_cv_image, "bgr8")
            #publish the image 
            self.pub_denoised_img.publish(denoised_ros_image_msg)

        except CvBridgeError as e:
            rospy.logerr(e)
        
def main():
    Denoise()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("shutting down ...")

if __name__ == "__main__":
    main()
    

    