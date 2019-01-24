if [ "$#" -ne 1 ]; then
    echo "Provide one command line argument"
    exit
fi
export PROJECT="protein229117"
export BUCKET="hpc_data"
export REGION="us-west1"
export MODEL_NAME="HPC"
export TRAINING_DIR="resnet50_trained"
export CUSTOM_JOB_NAME=$1
export TFVERSION='1.10'

OUTDIR=gs://${BUCKET}/${TRAINING_DIR}/${CUSTOM_JOB_NAME}
JOBNAME=${MODEL_NAME}_$(date -u +%y%m%d_%H%M%S)

# gsutil -m rm -rf $OUTDIR

gcloud ml-engine jobs submit training $JOBNAME \
   --region=$REGION \
   --module-name=trainer.task \
   --package-path=./trainer \
   --job-dir=$OUTDIR \
   --staging-bucket=gs://$BUCKET \
   --config ./trainer/training_config.yaml \
   --runtime-version=$TFVERSION \
   -- \
   --train_data_files="gs://${BUCKET}/data_extended/train*.tfrecord" \
   --eval_data_files="gs://${BUCKET}/data_extended/validate*.tfrecord"  \
   --test_data_files="gs://${BUCKET}/data_extended/test/test*.tfrecord"  \
   --model_dir=$OUTDIR \
   --batch_size=26 \
