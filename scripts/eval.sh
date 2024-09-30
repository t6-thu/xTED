logdirs=(
  ""
)

while getopts ":g:" opt; do
  case $opt in
    g)
      gpu_index=$OPTARG
      ;;
    \?)
      echo "Unknown options: -$OPTARG" >&2
      ;;
  esac
done

for logdir in "${logdirs[@]}"; do
    eval $"python evaluate.py $logdir --epochs 0 100 200 300 400 600 700 800 -g $gpu_index"
done