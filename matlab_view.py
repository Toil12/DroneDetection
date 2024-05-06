import scipy

annots=scipy.io.loadmat(f"dataset_local/5/anotation.mat")
# print(annots)
for key,value in annots.items():
    print(key,value)