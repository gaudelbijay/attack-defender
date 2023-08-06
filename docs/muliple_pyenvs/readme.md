ToDo

- create pyenv with different dependencies as needed.
- build the main catkin_ws with default python3 of the system.

- use the shebang `#!/usr/bin/env python3` in the python codes as the default of iros_image_attack pkg. 
- use the shebang `#!/home/$USER/mypyenvs/py310/bin/python` in the python scripts of our model, where we are referring to the pythonpath of the - virtual environment.

- make sure you put the name of the scripts with the shebang `#!/home/$USER/mypyenvs/py310/bin/python` into the CMakeLists.txt file as the example below:

```
catkin_install_python(PROGRAMS src/talker310.py src/talker36.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

here is what i've done:

```
mkdir -p ~/catkin_nuts/src 

catkin init
catkin create pkg pkg_310 roscpp rospy std_msgs # create a pkg with for python 3.10 with a venv 
catkin create pkg pkg_36 # create a pkg with for python 3.6 which is the system default
cd pkg_310/src
touch taker310.py
touch taker36.py
```
put the following scripts in each of the files `taker310.py` and 'taker36.py' with different shenags as stated above.

```
#!/home/$USER/mypyenvs/py310/bin/python
import numpy as np
import scipy
import rospy
from std_msgs.msg import Float64
print(np.__version__)
# license removed for brevity
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

make them executable with `chmod +x taker310.py` and `chmod +x taker36.py`.

now take care of `catkin_install_python()` as stated above. 

build the entire thing with `catkin_make`.

run `roscore`.

then run `rosrun pkg_py310 taker36.py`; it runs properly as

```
1.19.5
[INFO] [1691364186.527978]: hello world 1691364186.5277922
[INFO] [1691364186.628594]: hello world 1691364186.6283345
[INFO] [1691364186.728617]: hello world 1691364186.7283173
```

but runnning `rosrun pkg_py310 taker36.py` throws error for numpy and scipy because we did not install them in virtual pyenv3.10.

this example shows the indendence of ros nodes with two PYTHONPATHs.

Refs.

http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

http://wiki.ros.org/rospy_tutorials/Tutorials/Makefile

http://wiki.ros.org/rospy_tutorials/Tutorials/PythonPath
