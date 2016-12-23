#!/usr/bin/env sh
rm -rf ./MLP2_LittleBigData.zip
cd submission
git archive -o ../MLP3_LittleBigData.zip HEAD .
cd ../
git submodule init
git submodule update
python ./checkSub/checkSub.py MLP3_LittleBigData.zip

# TODO: adjust readme
# TODO: adjust final_sub.csv
# TODO: adjust src/classification.py
