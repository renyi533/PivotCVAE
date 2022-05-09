FAIL="0"
wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}

python preprocess_ml1m.py

python pretrain_env.py --dataset movielens --s 5 --dim 8 --resp_struct [48,256,256,5] --epoch 10 --batch_size 64 --lr 0.001 --wdecay 0.001 --device cuda:0


wait_function

j="0"
for lr in 0.001 0.0003 0.002 0.00015
do

  for wdecay in 1e-4 1e-5 1e-6 1e-7 1e-3 1e-2
  do 
  python train_deterministic.py --dataset movielens --resp_path resp/movielens/resp_[48,256,256,5]_dim8__BS64_lr0.00100_decay0.00100 --model neumf --dim 8 --s 5 --batch_size 64 --lr $lr --wdecay $wdecay --device cuda:0 --nneg 50 --struct [16,256,256,1] --epoch 35 --early_stop 0 > neumf_lr${lr}_wdecay${wdecay}.log 2>&1 &

    for weight in 0.0625 0.125  0.25  0.5  1.0  2.0  4.0  8.0 16.0 32.0 64.0 128.0 
    do
      python train_deterministic.py --dataset movielens --resp_path resp/movielens/resp_[48,256,256,5]_dim8__BS64_lr0.00100_decay0.00100 --model neumf_cvae --dim 8 --s 5 --batch_size 64 --lr $lr --wdecay $wdecay --device cuda:0 --nneg 50 --struct [32,256,256,1] --epoch 35 --early_stop 0 --mse_weight $weight > cvae_lr${lr}_wdecay${wdecay}_lossweight${weight}.log 2>&1 &


      python train_deterministic.py --dataset movielens --resp_path resp/movielens/resp_[48,256,256,5]_dim8__BS64_lr0.00100_decay0.00100 --model pfd --dim 8 --s 5 --batch_size 64 --lr $lr --wdecay $wdecay --device cuda:0 --nneg 50 --struct [16,256,256,1] --epoch 35 --early_stop 0 --mse_weight $weight > pfd_lr${lr}_wdecay${wdecay}_lossweight${weight}.log 2>&1 &

      j=$[$j+2]

      n=$(($j%4))

      echo $n

      if [ $n -eq 0 ]; then wait_function; fi

    done
  done
done

wait_function