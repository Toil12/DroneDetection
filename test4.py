data=['B03','B02','B01','A02','A01','A03',]
import re
pattern=''
array=[]
for item in data:
    temp=[]
    temp.append(re.split('\d',item)[0])
    temp.append(re.split('\D',item)[1])
    array.append(temp)
array=sorted(array,key=(lambda x:[x[0],x[1]]))
results=[]
cache=[]
tag=''
for i in range(len(array)):
    if cache==[]:
        cache.append(array[i][1])
        tag = array[i][0]
    elif array[i][0]!=tag or int(array[i][1])-int(cache[-1])!=1 or i==len(array)-1:
        if i==len(array)-1:
            cache.append(array[i][1])
            result_temp = f"{tag}{cache[0]}-{cache[-1]}"
            results.append(result_temp)
            break
        result_temp=f"{tag}{cache[0]}-{cache[-1]}"
        results.append(result_temp)
        cache=[array[i][1]]
        tag=array[i][0]
    else:
        cache.append(array[i][1])

print(results)