cd /data/deep_server/build
nohup /data/deep_server/build/deep_server --port=6666 -hm -l deep.log --lib_mode=caffe --batch_size=1 --gpu=2,2 --caffe_weights=../model/frcnn_prune/vgg16_faster_rcnn_2class_prune_iter_70000.caffemodel --caffe_default_c=../model/frcnn_prune/voc_config.json --caffe_model=../model/frcnn_prune/test.prototxt &
