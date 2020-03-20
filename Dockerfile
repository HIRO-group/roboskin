FROM ros:melodic
# install the necessary packages
RUN apt-get update

COPY . .
