#!/bin/bash

# Get files from Google Drive

id=1Egx0t5bWMLvcHpLYP5HqptRyCh1Mj_S- # change id when new baseline is uploaded
name=baseline

URL="https://docs.google.com/uc?export=download&id=$id"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=$id" -O $name && rm -rf /tmp/cookies.txt}
