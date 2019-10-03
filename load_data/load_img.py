import urllib.request
import pandas as pd
import os

#set necessary file path
#path_file= "/home/ubuntu/Deep-Learning/load_data"
#path_dir= "/home/ubuntu/Deep-Learning/load_data/test/{}"
path_file= '/Users/binbinwu/Desktop/Capstone/Deep-Learning/load_data'
path_dir_test= '/Users/binbinwu/Desktop/Capstone/Deep-Learning/load_data/test/{}'
path_dir_train= '/Users/binbinwu/Desktop/Capstone/Deep-Learning/load_data/train/{}'
#path_dir_validation= '/Users/binbinwu/Desktop/Capstone/Deep-Learning/load_data/validation/{}'

#set url opener
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)


def downloader(image_url,file_name,count,dir_path):
    full_file_name = str(file_name) + '.' + str(count) + '.jpg'
    dir_file_name = os.path.join(dir_path, full_file_name)
    urllib.request.urlretrieve(image_url,dir_file_name)

#create list of files
numFiles = []
fileNames = os.listdir(path_file)
for fileNames in fileNames:
    if fileNames.endswith(".csv"):
        numFiles.append(fileNames)


#url=pd.read_csv('/home/ubuntu/Deep-Learning/load_data/train/Taraxacum officinale/observations-63603.csv')
for file in numFiles:
    url = pd.read_csv(file)
    dir_name=url['scientific_name'][0]
    os.makedirs(path_dir_train.format(dir_name), mode=0o777, exist_ok=True)
    os.makedirs(path_dir_test.format(dir_name), mode=0o777, exist_ok=True)

    for i in range(10):
        try:
            downloader(url['image_url'][i], url['scientific_name'][i], i, path_dir_train.format(dir_name))
        except Exception as e:
            print(e)
            continue
    for i in range(11,20):
        try:
            downloader(url['image_url'][i], url['scientific_name'][i], i, path_dir_test.format(dir_name))
        except Exception as e:
            print(e)
            continue



