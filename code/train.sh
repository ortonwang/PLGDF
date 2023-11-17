
# train
python train.py --dataset_name Pancreas_CT --model VNet_4out --exp v1 --labelnum 6 --gpu 1
python train.py --dataset_name Pancreas_CT --model VNet_4out --exp v1 --labelnum 12 --gpu 0

python train.py --dataset_name LA --model VNet_4out --exp v1 --labelnum 8 --gpu 0
python train.py --dataset_name LA --model VNet_4out --exp v1 --labelnum 16 --gpu 0




#test
python test_3d.py --dataset_name Pancreas_CT --model VNet_4out --exp v1 --labelnum 6 --gpu 0
python test_3d.py --dataset_name Pancreas_CT --model VNet_4out --exp v1 --labelnum 12 --gpu 0

python test_3d.py --dataset_name LA --model VNet_4out --exp v1 --labelnum 8 --gpu 0
python test_3d.py --dataset_name LA --model VNet_4out --exp v1 --labelnum 16 --gpu 0