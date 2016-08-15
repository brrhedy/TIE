#! /bin/sh
predict_month=$1
begin=$2
end=$3

if [ $# -lt 3 ]; then 
    echo 'run_RF_train_L1.sh Parameters: predict_month begin end'
    exit 1
fi

rm -r model_RF/${predict_month}
mkdir model_RF/${predict_month}/classifier
mkdir model_RF/${predict_month}/regressor

for((i=${begin};i<=${end};i++))
do 
    echo 'the trainning data is: train_data_'${i}'.csv'

    mkdir model_RF/${predict_month}/classifier/rf_${i}_300t
    mkdir model_RF/${predict_month}/regressor/rf_${i}_300t
    python src/rf_L1_train.py train_data/${predict_month}/train_data_${i}.csv  model_RF/${predict_month}/classifier/rf_${i}_300t/rf_${i}_300t.md model_RF/${predict_month}/regressor/rf_${i}_300t/rf_${i}_300t.md feature_d.txt importances/${predict_month}/importance_${i}.txt
done
