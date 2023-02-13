from utils.autoanchor import polygon_kmean_anchors

nl = 3 # number of anchor layers
na = 3 # number of anchors
img_size = 640 # image size for training and testing

datacfg = "/home2/shaon/PolygonObjectDetection/polygon-yolov5/data/polygon_ucas.yaml"
anchors = polygon_kmean_anchors(datacfg, n=nl*na, gen=1000, img_size=img_size)
print(anchors.reshape(nl, na*2).astype(int))
print('\nPlease Copy the anchors to your model configuration polygon_yolov5*.yaml')