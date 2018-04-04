
import cv2
for i in range(1,11):
    S="Img_org\cellphone ("+str(i)+").jpg"
    img_org = cv2.imread(S)
    img = cv2.resize(img_org,(512,320))
    rS="Img_Data\dataImg"+str(i)+".jpg"
    cv2.imwrite(rS,img)
    print("resized Img: "+S)

