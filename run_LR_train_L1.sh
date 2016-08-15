#! /bin/sh
predict_month=$1
begin=$2
end=$3

if [ $# -lt 3 ]; then 
    echo 'run_LR_train_L1.sh Parameters: predict_month begin end'
    exit 1
fi

rm -r model_LR/${predict_month}
mkdir model_LR/${predict_month}

for((i=${begin};i<=${end};i++))
do 
    echo 'the trainning data is: train_data_'${i}'.csv'    

    mkdir model_LR/${predict_month}/lr_${i}
    python src/lr_L1_train.py train_data/${predict_month}/train_data_${i}.csv model_LR/${predict_month}/lr_${i}/lr_${i}.md      
done

