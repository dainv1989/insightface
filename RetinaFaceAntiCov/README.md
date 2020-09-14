Face demo  

* detect quay trái/phải  
* detect mắt (nhắm mắt) sử dụng dlib shape detector và face_utils  
* detect khẩu trang (insightface)  

cài đặt rcnn cho RetinaFaceAntiCov  
```shell script
cp -rf ../RetinaFace/rcnn/* ./rcnn/
python setup.py build_ext --inplace 
```

requirements:  
```shell script
dlib
imutils
opencv-python
```

tải trước pretrained model bỏ vào thư mục ``model``  