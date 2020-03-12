FROM ros:melodic
# install the necessary packages
RUN apt-get update && apt-get install -y python3-pip python3-yaml \
    && sudo pip3 install rospkg catkin_pkg

COPY . .
