import urllib.request
import pandas as pd

def downloader(image_url,file_name,count):
    full_file_name = str(file_name) + str(count)+'.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


url=pd.read_csv('/home/ubuntu/Deep-Learning/load_data/test.csv')

for i in range(len(url)):
    downloader(url['image_url'][i],url['common_name'][i],i)






#with open ('/home/ubuntu/Deep-Learning/load_data/test.csv') as images:

