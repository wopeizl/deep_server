#!/bin/sh

checkprocess() {
       if [ "$1" = "" ];then
       	   return 1
       fi

       pid=$(pgrep "$1");
       if [ -n "$pid" ]; then
           return 0 
       else
           return 1
       fi
}

while [ 1 ] ; do
       checkprocess "deep_server"
       check_result=$?
   if [ $check_result -eq 1 ];
       then
           sh control.sh restart
   fi
       sleep 3
done

