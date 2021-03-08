now=0.1_16
mkdir ./SubCLTS/less7000/$now/
cp ./SubCLTS/lcsts.yaml ./SubCLTS/less7000/$now/lcsts.yaml
nohup python train.py -gpus 1 -config ./SubCLTS/less7000/$now/lcsts.yaml -log ./SubCLTS/less7000/$now/ -nxpath $now > ./SubCLTS/less7000/$now/less566 2>&1 &
