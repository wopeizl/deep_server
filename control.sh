#!/bin/bash
#
# description: deep_server init script
# processname: deep_server
# chkconfig: 
  
start() {
  cd /data/deep_server/build
  nohup /data/deep_server/build/deep_server --port=6666 -hm -l deep.log --lib_mode=caffe --batch_size=1 --gpu=2,2 --caffe_weights=../model/frcnn_prune/vgg16_faster_rcnn_2class_prune_iter_70000.caffemodel --caffe_default_c=../model/frcnn_prune/voc_config.json --caffe_model=../model/frcnn_prune/test.prototxt &
  echo "deep_server is started"
}
 
stop() {
  pid=$(pgrep deep_server)
  echo "deep_server pid=$pid"
  if [ -n "$pid" ]; then
    kill -9 "$pid"
  fi
  echo "deep_server is stopped"
}

status(){
  pid=$(pgrep deep_server)
  echo "deep_server pid=$pid"
  if [ -n "$pid" ]; then
    echo "deep_server is running"
  else
    echo "deep_server is stopped"
  fi

}
 
case $1 in
        start)
          start
        ;;
        stop)  
          stop
        ;;
        restart)
          stop
          start
        ;;
        status)
                status
                exit $?  
        ;;
        kill)
                terminate
        ;;
        *)
                echo -e "no parameter"
        ;;
esac    
exit 0
