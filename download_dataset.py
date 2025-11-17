import kagglehub

# Download latest version
path = kagglehub.dataset_download("fareselmenshawii/license-plate-dataset")

print("Path to dataset files:", path)