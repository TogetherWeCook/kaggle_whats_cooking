if [ ! -d "data" ]; then
  mkdir data
fi
kaggle competitions download -c whats-cooking-kernels-only -p data