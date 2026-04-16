FROM osrf/ros:humble-desktop

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Yiyu: Install packages ========================================
# Installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    iproute2 \
    libasio-dev \
    udev \
&& apt-get install -y \
    python3-pip \
&& rm -rf /var/lib/apt/lists/*
# Removed Packages: 
    # ros-$ROS_DISTRO-rplidar-ros \
    # gazebo \
    # libgazebo-dev \
    # gedit \
    # ros-$ROS_DISTRO-velodyne \
    # ros-$ROS_DISTRO-pointcloud-to-laserscan \
    # ros-$ROS_DISTRO-realsense2-camera \
    # ros-$ROS_DISTRO-rtabmap-ros \

#https://github.com/2b-t/velodyne-ros2-docker/blob/main/docker/Dockerfile

# Yiyu: Install packages ========================================

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME
RUN sudo usermod --append --groups plugdev $USERNAME

# Update all packages
# RUN sudo apt update && sudo apt upgrade -y

# Rosdep update
# RUN rosdep update

# WORKDIR /RTAB_Nav2_ArUco

# Install python packages
# COPY requirements.txt /ros2_ws/Docker/requirements.txt
# RUN pip install -r /ros2_ws/Docker/requirements.txt
RUN python3 -m pip install --no-cache-dir msgpack 'websockets>=11.0'

# bashrc: Source the ROS setup file and use color prompt
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc && \
    sed -i -e 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc
