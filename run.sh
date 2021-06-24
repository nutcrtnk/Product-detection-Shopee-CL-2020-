data=train
ratio=0.95
options="-m efn_b5_ns  --batch_size 32 -p ${data} --train_ratio ${ratio}"
im1=456
im2=600
m1=efn_b5_${im1}
m2=${m1}_${im2}

python split.py -p ${data} --train_ratio ${ratio}
python train.py  --img_size ${im1} --out ${m1}  ${options}
python train.py --img_size ${im2} --load_ckpt ${m1}.pt --out ${m2} --options "" ${options}
python test.py -i ${data}_${ratio}/${m2}.pt  --img_size ${im2}