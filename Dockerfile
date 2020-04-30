FROM ros:melodic
# install the necessary packages
RUN apt update && apt -y install python3-pip && mkdir robotic_skin
COPY . robotic_skin/
RUN cd robotic_skin && pip3 install .
