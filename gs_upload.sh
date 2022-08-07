# Usage : bash gs_upload.sh YEAR
for m in {1..9}
do
  gsutil -m cp $1_images/$1_0${m}_* gs://medium_data/$1_images
done

for m in 0 1 2
do
  gsutil -m cp $1_images/$1_1${m}_* gs://medium_data/$1_images
done

gsutil cp $1_medium_data.csv gs://medium_data/
