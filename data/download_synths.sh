# Need gsutil installed. run `conda install conda-forge::gsutil` to install it or manually download datasets from https://console.cloud.google.com/storage/browser/multi-object-datasets;tab=objects?pli=1&inv=1&invt=AbjJBg&prefix=&forceOnObjectsSortingFiltering=false
for dataset in tetrominoes multi_dsprites clevr_with_masks; do
    gsutil cp -r gs://multi-object-datasets/$dataset ./
    python convert_tfrecord_to_np.py --dataset_name=$dataset
done
