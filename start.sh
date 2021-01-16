#!/bin/bash

FILE=../elliptic_bitcoin_dataset/full_data.csv
if [[ -f "$FILE" ]]; then
    echo "$FILE Exist"
else
    echo "Adding Features"
    cd orginizing_data_src/
    python3 connect_real_txs.py
    cd ..
fi
case $1 in
    -all|-A)
        cd raw_ML/
        bash run_list.sh
        cd ../modified_ML
        bash run_list.sh
        cd ../pca_ML
        bash run_list.sh
        cd ../feature_select_corr_ML
        bash run_list.sh
        cd ../feature_select_pca_ML/
        bash run_list.sh
        cd ../stat
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    -raw|-R)
        cd raw_ML/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    -modified|-M)
        cd modified_ML/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    -pca|-P)
        cd pca_ML/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    pcaf|-PF)
        cd feature_select_pca_ML/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    -corrf|-CF)
        cd feature_select_corr_ML/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    -stat|-S)
        cd stat/
        bash run_list.sh
        cd ..
        echo "Finished $1"
    ;;
    *)
        echo "Wrong Flag"
    ;;
esac