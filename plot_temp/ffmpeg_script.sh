#!/bin/bash
echo "Creating mp4 from list of images..."
ffmpeg -framerate 20 -i sphere_test%d.png sphere_anim_test.mp4
