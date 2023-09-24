#!/usr/bin/env bash
# https://vitorbaptista.com/how-to-access-hosts-docker-socket-without-root

DOCKER_SOCKET=/var/run/docker.sock
DOCKER_GROUP=docker

if [ -S ${DOCKER_SOCKET} ]; then
    
    DOCKER_GID=$(stat -c '%g' ${DOCKER_SOCKET})
    if ! grep -q ${DOCKER_GROUP} /etc/group; then
        groupadd -g ${DOCKER_GID} ${DOCKER_GROUP}
    fi
    # Add user to group
    usermod -aG ${DOCKER_GROUP} ${SUDO_USER}
fi