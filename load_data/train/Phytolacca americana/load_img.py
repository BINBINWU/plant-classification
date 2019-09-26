import urllib.request
import pandas as pd

opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

def downloader(image_url,file_name,count):
    full_file_name = str(file_name) + '.'+str(count)+'.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


url=pd.read_csv('/home/ubuntu/Deep-Learning/load_data/train/Phytolacca americana/observations-64147.csv')

for i in range(500):
    try:
        downloader(url['image_url'][i],url['scientific_name'][i],i)
    except Exception as e:
        print (e)
        continue







#with open ('/home/ubuntu/Deep-Learning/load_data/test.csv') as images:

