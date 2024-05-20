環境:
python 3.7
numpy
cv2

1.創建一個新資料夾在TestImage內，新資料內放入將不同曝光時間的照片
2.新增一個image_list.txt在照片旁邊
image_list.txt 內寫入對應的圖片檔名與曝光時間(s)

例如:
# Filename	exposure
result_4.jpg	0.0625
result_5.jpg	0.03125
result_6.jpg	0.015625
result_7.jpg	0.0078125

3.在code folder內打開PSF_HDRI_maker.py
4.將第7行的TestImage = 'metalens_PSF'改為新建立的資料夾名稱
5.執行
6.HDR影像會儲存在Result內