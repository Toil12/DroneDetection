import scipy

annots=scipy.io.loadmat(rf"F:\Data\USC\youtube\image&label-20240421T165405Z-001\image&label\2\anotation")
# print(annots)
for key,value in annots.items():
    print(key,value)