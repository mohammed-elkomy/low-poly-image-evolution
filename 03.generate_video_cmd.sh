#####################################################################################
############################## Generate video demo ##################################
#####################################################################################
ffmpeg -r 60 -f image2  -s 683x1024 -i %05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" test.mp4

# test.mp4 is the output video name
# %05d.png the input images matching wildcard
# 60 is frame rate