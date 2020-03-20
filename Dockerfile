FROM ros:melodic
# install the necessary packages
RUN apt update && apt -y install python-pip && mkdir robotic_skin
COPY . robotic_skin/
RUN cd robotic_skin && pip install .
