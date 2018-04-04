# Img2Data
This repository haves some python files to make images into data about labelme.
1. resize all images in Img_org ,than save into Img_Data with new names.
python resize_rename.py

2. using labelme to mark your images,than using jsontodata.py to get your data.
python jsontodata.py

3. change the 16bits label.png into 8bits label.png
python cv16to8.py
