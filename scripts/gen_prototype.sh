#!/bin/bash

# custom config
DATA=/home/longhorn/wushihan/Datasets/data/
TRAINER=DAPT

DEVICE_OFFSET=2

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101
# for DATASET in imagenet
    do
    # for SHOTS in 1 2 4 8 16
    for SHOTS in 16
    do
        if [ ${DATASET} == "imagenet" ]; then
            CFG=vit_b16_ep50
        elif [ ${SHOTS} -eq 1 ]; then
            CFG=vit_b16_ep50
        elif [ ${SHOTS} -eq 2 ] || [ ${SHOTS} -eq 4 ]; then
            CFG=vit_b16_ep100
        elif [ ${SHOTS} -eq 8 ] || [ ${SHOTS} -eq 16 ]; then
            CFG=vit_b16
        fi

        for SEED in 1 2 3
        do
            sleep $DEVICE_OFFSET

            DEVICE_ID=$(( $SEED + $DEVICE_OFFSET ))
            (
                CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
                python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                DATASET.NUM_SHOTS ${SHOTS} \
                TRAINER.DAPT.PROTOTYPE_GEN True
            ) &
        done
        wait
    done
done
