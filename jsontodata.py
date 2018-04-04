import os
s1="labelme_json_to_dataset Img_Data\dataImg"
s3=".json"

for i in range(1,11):
    s2=str(i)
    os.system(s1+s2+s3)
