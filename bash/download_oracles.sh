#!/bin/bash

wget https://zenodo.org/records/20025380/files/oracles.zip
OUT=oracles.zip
DEST=objectives
mkdir -p "$DEST"

unzip $OUT -d $DEST
rm $OUT