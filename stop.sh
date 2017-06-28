pid=$(pgrep deep_server)
echo "deep_server pid=$pid"
if [ -n "$pid" ]; then
    kill -9 "$pid"
fi
