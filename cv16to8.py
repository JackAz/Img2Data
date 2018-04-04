from PIL import Image

for i in  range(1,11):
    S = "Img_Data\dataImg"+str(i)+"_json\label.png"
    I = Image.open(S)
    L = I.convert('L')
   # L.show()
    L.save(S)
    print("save image: "+S)
#I.show()
#L = I.convert('L')   #转化为灰度图
#L = I.convert('1')   #转化为二值化图
#L.resize((800,800))


#import cv2
#import numpy as np
#img_org = cv2.imread('cellphone(1).jpg')
#img = cv2.resize(img_org,(600,400))
#img=cv2.cvtColor(img,1)

#cv2.imshow('Cell Phone',img)
#cv2.imwrite('imwrite.png', L)
#等待输入
#cv2.waitKey()
#cv2.destroyAllWindows()