# syntax=docker/dockerfile:1

ARG ROS_DISTRO=noetic
ARG ROS_IMAGE=osrf/ros:noetic-desktop-full

# Build the non-apt C++ dependencies separately so Docker can cache them when
# only the ROS workspace changes.
FROM ${ROS_IMAGE} AS third-party-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG CASADI_VERSION=3.5.5
ARG LIVOX_SDK_VERSION=v1.3.1
ARG NANOFLANN_COMMIT=fd8cf41ff6b321c4f96a3eb44f4b9c92cf2163d8
ARG OPEN3D_VERSION=0.15.1
ARG OSQP_VERSION=v0.6.3

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      coinor-libipopt-dev \
      curl \
      gfortran \
      git \
      liblapack-dev \
      pkg-config \
      xz-utils && \
    rm -rf /var/lib/apt/lists/*

# CasADi supplies the C++ API used by nmpc_planner. A Python wheel alone is
# not ABI-compatible with this workspace's C++ nodes.
RUN git clone --depth 1 --branch "${CASADI_VERSION}" \
      https://github.com/casadi/casadi.git /tmp/casadi && \
    cmake -S /tmp/casadi -B /tmp/casadi/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/casadi \
      -DWITH_EXAMPLES=OFF \
      -DWITH_IPOPT=ON \
      -DWITH_PYTHON=OFF && \
    cmake --build /tmp/casadi/build --parallel "$(nproc)" && \
    cmake --install /tmp/casadi/build

RUN git clone --depth 1 --branch "${OSQP_VERSION}" --recurse-submodules \
      https://github.com/osqp/osqp.git /tmp/osqp && \
    cmake -S /tmp/osqp -B /tmp/osqp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DUNITTESTS=OFF && \
    cmake --build /tmp/osqp/build --parallel "$(nproc)" && \
    cmake --install /tmp/osqp/build

RUN git init /tmp/nanoflann && \
    git -C /tmp/nanoflann remote add origin \
      https://github.com/jlblancoc/nanoflann.git && \
    git -C /tmp/nanoflann fetch --depth 1 origin "${NANOFLANN_COMMIT}" && \
    git -C /tmp/nanoflann checkout --detach FETCH_HEAD && \
    cmake -S /tmp/nanoflann -B /tmp/nanoflann/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DNANOFLANN_BUILD_EXAMPLES=OFF \
      -DNANOFLANN_BUILD_TESTS=OFF && \
    cmake --build /tmp/nanoflann/build --parallel "$(nproc)" && \
    cmake --install /tmp/nanoflann/build

RUN git clone --depth 1 --branch "${LIVOX_SDK_VERSION}" \
      https://github.com/Livox-SDK/Livox-SDK2.git /tmp/livox-sdk2 && \
    cmake -S /tmp/livox-sdk2 -B /tmp/livox-sdk2/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/livox-sdk2/build --parallel "$(nproc)" && \
    cmake --install /tmp/livox-sdk2/build

# The project's open3d_loc package is tied to Open3D 0.15.1 and the current
# machine is amd64. This official development archive contains the C++ ABI
# used by the working native installation.
RUN test "$(dpkg --print-architecture)" = "amd64" || \
      { echo "The full profile currently requires the amd64 Open3D archive." >&2; exit 1; } && \
    mkdir -p /opt/open3d && \
    curl -fsSL \
      "https://github.com/isl-org/Open3D/releases/download/v${OPEN3D_VERSION}/open3d-devel-linux-x86_64-cxx11-abi-${OPEN3D_VERSION}.tar.xz" \
      | tar -xJ --strip-components=1 -C /opt/open3d


FROM ${ROS_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=noetic
ARG BUILD_PROFILE=all
ARG UNITREE_SDK2PY_COMMIT=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NAV3D_WS=/opt/nav3d_ws \
    NAV3D_DATA_ROOT=/data \
    CASADI_ROOT=/opt/casadi \
    CASADI_LIB_PATH=/opt/casadi/lib \
    Open3D_DIR=/opt/open3d/lib/cmake/Open3D \
    LD_LIBRARY_PATH=/opt/nav3d_ws/devel/lib:/opt/casadi/lib:/opt/open3d/lib:/usr/local/lib:/opt/ros/noetic/lib

# Keep build tools in the final image: this is a complete ROS development
# environment, so the mounted workspace can be rebuilt inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      coinor-libipopt1v5 \
      curl \
      git \
      iproute2 \
      libapr1-dev \
      libc++-dev \
      libc++abi-dev \
      libeigen3-dev \
      libgoogle-glog-dev \
      libgflags-dev \
      libgl1 \
      liblapack3 \
      libnlopt-cxx-dev \
      libopencv-dev \
      libpcap-dev \
      libpcl-dev \
      libtbb-dev \
      libudev1 \
      libx11-6 \
      libyaml-cpp-dev \
      net-tools \
      pkg-config \
      python3-catkin-tools \
      python3-dev \
      python3-matplotlib \
      python3-numpy \
      python3-pip \
      python3-rosdep \
      python3-rospkg \
      python3-scipy \
      python3-yaml \
      ros-${ROS_DISTRO}-eigen-conversions \
      ros-${ROS_DISTRO}-geometry-msgs \
      ros-${ROS_DISTRO}-message-generation \
      ros-${ROS_DISTRO}-message-runtime \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-octomap-msgs \
      ros-${ROS_DISTRO}-octomap-ros \
      ros-${ROS_DISTRO}-pcl-conversions \
      ros-${ROS_DISTRO}-pcl-ros \
      ros-${ROS_DISTRO}-pluginlib \
      ros-${ROS_DISTRO}-rosbag \
      ros-${ROS_DISTRO}-roscpp \
      ros-${ROS_DISTRO}-rospy \
      ros-${ROS_DISTRO}-rviz \
      ros-${ROS_DISTRO}-sensor-msgs \
      ros-${ROS_DISTRO}-std-msgs \
      ros-${ROS_DISTRO}-tf \
      ros-${ROS_DISTRO}-tf2 \
      ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-tf2-ros && \
    rm -rf /var/lib/apt/lists/*

COPY --from=third-party-builder /opt/casadi /opt/casadi
COPY --from=third-party-builder /opt/open3d /opt/open3d
COPY --from=third-party-builder /usr/local/ /usr/local/
RUN ldconfig

# Match the Python side of the native Go2 environment. ROS Python packages
# remain provided by apt; these packages provide Unitree DDS communication.
RUN python3 -m pip install --no-cache-dir \
      casadi==3.7.2 \
      cyclonedds==0.10.2 \
      numpy==1.24.4 \
      opencv-python==5.0.0.93 \
      scipy==1.10.1

RUN mkdir -p /tmp/unitree-sdk2py && \
    curl -fsSL --retry 5 --retry-delay 3 --connect-timeout 30 --max-time 300 \
      "https://codeload.github.com/unitreerobotics/unitree_sdk2_python/tar.gz/${UNITREE_SDK2PY_COMMIT}" \
      | tar -xz --strip-components=1 -C /tmp/unitree-sdk2py && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/unitree-sdk2py

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-sophus && \
    rm -rf /var/lib/apt/lists/*

COPY . ${NAV3D_WS}
WORKDIR ${NAV3D_WS}

# Package manifests contain several non-standard rosdep keys for Eigen, so
# the known dependencies are installed explicitly above. rosdep is still used
# to fill any valid package dependencies added later.
RUN rosdep install --from-paths src --ignore-src -r -y \
      --skip-keys="Eigen eigen3 libeigen3-dev libpcl-all libpcl-all-dev open3d osqp casadi" && \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
    case "${BUILD_PROFILE}" in \
      all) \
        catkin_make -DCMAKE_BUILD_TYPE=Release \
          -DOpen3D_DIR="${Open3D_DIR}" \
          -DCATKIN_WHITELIST_PACKAGES="" \
        ;; \
      core|mid360) \
        ./scripts/build_workspace.sh "${BUILD_PROFILE}" \
        ;; \
      *) \
        echo "BUILD_PROFILE must be one of: all, mid360, core" >&2; exit 2 \
        ;; \
    esac && \
    ldconfig

RUN mkdir -p "${NAV3D_DATA_ROOT}" && \
    chmod +x docker/entrypoint.sh

VOLUME ["/data"]
ENTRYPOINT ["/opt/nav3d_ws/docker/entrypoint.sh"]
CMD ["bash"]
