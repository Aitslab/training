install scrapy in a new conda environment: 
conda install -c conda-forge scrapy

navigate to folder of choice in which you want to have the spider and create a projectfolder with:
scrapy startproject yourprojectname

if you have  AttributeError: type object 'SettingsFrame' has no attribute 'ENABLE_CONNECT_PROTOCOL' update h2 and try again:
 run pip3 install --upgrade h2
 
navigate to spider subfolder in project folder and create a spider file yourspidername.py

edit yourspidername.py as needed including a spider name, see here for more guidance: https://docs.scrapy.org/en/latest/intro/tutorial.html

run spider in command prompt with
scrapy crawl spidername

