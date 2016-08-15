#! /bin/sh
predict_month=$1
begin=$2
end=$3


hadoop fs -get /user/bairr/TIE/data

bash run_RF_train_L1.sh ${predict_month} ${begin} ${end}
bash run_LR_train_L1.sh ${predict_month} ${begin} ${end}
bash run_SVM_train_L1.sh ${predict_month} ${begin} ${end}


mkdir L1_result/rf_result/classifier/${predict_month}
mkdir L1_result/rf_result/regressor/${predict_month}
mkdir L1_result/lr_result/classifier/${predict_month}
mkdir L1_result/lr_result/regressor/${predict_month}
mkdir L1_result/svm_result/classifier/${predict_month}
mkdir L1_result/svm_result/regressor/${predict_month}


begin_month=${predict_month}
for (( i = 0; i < 4; i++ )); 
do
	if i == 0; then
		bash run_predict_L1.sh ${predict_month} ${begin} ${end} ${begin_month} predict
	else
		bash run_predict_L1.sh ${predict_month} ${begin} ${end} ${begin_month} train
	fi

    nohup hadoop fs -put L1_result/rf_result/classifier/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/RF/classifier/${predict_month}/${begin_month}/ &
    nohup hadoop fs -put L1_result/rf_result/regressor/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/RF/regressor/${predict_month}/${begin_month}/ &
    nohup hadoop fs -put L1_result/lr_result/classifier/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/LR/classifier/${predict_month}/${begin_month}/ &
    nohup hadoop fs -put L1_result/lr_result/regressor/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/LR/regressor/${predict_month}/${begin_month}/ &
    nohup hadoop fs -put L1_result/svm_result/classifier/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/SVM/classifier/${predict_month}/${begin_month}/ &
    nohup hadoop fs -put L1_result/svm_result/regressor/${predict_month}/${begin_month}/${category}_${begin_month}_*_feature.csv /user/bairr/TIE/L1_RESULT/SVM/regressor/${predict_month}/${begin_month}/ &

	begin_month=`date -d "${begin_month}01 -1 month" +%Y%m%d`
done
