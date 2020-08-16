FROM ros:melodic
# install the necessary packages
RUN apt update && apt -y install python3-pip && mkdir roboskin
COPY . roboskin/
RUN cd roboskin && pip3 install .
