#! /bin/sh
predict_month=$1
begin=$2
end=$3

if [ $# -lt 3 ]; then 
    echo 'run_RF_train_L1.sh Parameters: predict_month begin end'
    exit 1
fi

rm -r model_linearSVM/${predict_month}
mkdir model_linearSVM/${predict_month}/regressor
mkdir model_linearSVM/${predict_month}/classifier

for((i=${begin};i<=${end};i++))
do 
    echo 'the trainning data is: train_data_'${i}'.csv'

    mkdir model_linearSVM/${predict_month}/classifier/svm_${i}
    mkdir model_linearSVM/${predict_month}/regressor/svm_${i}
    python src/linearSVM_L1_train.py train_data/${predict_month}/train_data_${i}.csv model_linearSVM/${predict_month}/classifier/svm_${i}/svm_${i}.md model_linearSVM/${predict_month}/regressor/svm_${i}/svm_${i}.md
done

