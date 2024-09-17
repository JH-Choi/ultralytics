
    # "Okutama-D-D1M-D1M.yaml" 
    # "Okutama-D-D2M-D2M.yaml" 
DATA=(
    "Okutama-D-D1N-D1N.yaml" 
    "Okutama-D-D2N-D2N.yaml" 
    "Okutama-D-D1D2M-D1D2M.yaml" 
    "Okutama-D-D1D2N-D1D2N.yaml" 
    "Okutama-D-ALL-ALL.yaml" 
) 

    # "Detect_D1M_val_D1M"
    # "Detect_D2M_val_D2M"
NAME=(
    "Detect_D1N_val_D1N"
    "Detect_D2N_val_D2N"
    "Detect_D1D2M_val_D1D2M"
    "Detect_D1D2N_val_D1D2N"
    "Detect_ALL_val_ALL"
)

length=${#DATA[@]}

for (( i=0; i<$length; i++ ))
do
   echo "DATA ${DATA[$i]} | NAME ${NAME[$i]}"
   python train.py --data "${DATA[$i]}" --name "${NAME[$i]}"
done