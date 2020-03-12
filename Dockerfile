FROM ros:melodic
# place here your application's setup specifics
RUN apt-get update && apt-get install -y python3-pip python3-yaml \
    && sudo pip3 install rospkg catkin_pkg \
    && git clone 

