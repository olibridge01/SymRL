#! /bin/bash

echo "<<UPDATING DOCKER IMAGE FOR SYMRL...>>"
docker build -f $RN_SOURCE_DIR/docker/psn/Dockerfile -t relnet/psn --rm $RN_SOURCE_DIR && docker image prune -f