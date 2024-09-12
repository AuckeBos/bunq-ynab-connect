#!/usr/bin/env bash
# temporary separate entrypoint for mlserver container
# we need to have a new version of mlserver, because old versions dont work with pydantic 2.0
# however, new version depends on old version of fastapi, which will disallow prefect >=3.0
# Because we want to keep using one Dockerfile, we temporarily (untill mlserver is update) update the config using this script
# It will uninstall prefect (not used by mlserver) ,and upgrade mlserver

# remove prefect
pip uninstall -y prefect
# update mlserver
pip install --upgrade mlserver
# start mlserver
mlserver start /home/bunqynab/config/mlserver