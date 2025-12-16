config_path=configs/CIFAR10/
length=${#config_path}

for f in $config_path*; do 
    t=${f[@]: $length:-5}
    echo $t
    python3 models_delta_visualize.py --config-name=$t --config-path=$config_path
done;
