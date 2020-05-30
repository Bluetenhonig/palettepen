# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:26:05 2020

@author: Linda Samsinger

Download all pins from all boards from a user's account
on Pinterest. 
"""

#TODO: search on pinterest, download first-n images

import requests
from bs4 import BeautifulSoup
import json

# get all pinterst boardnames of username 
username = "lindarden" 
account = "https://www.pinterest.ch/lindarden"
page=requests.get(account)
soup = BeautifulSoup(page.content, 'html5lib')
s = soup.find_all("script", {"id": "initial-state"})
data = soup.select("[type='application/json']")[1]
json_meta = json.loads(data.text)

# find metatags xpath in json 
xpath_root = json_meta['resourceResponses'][0]['response']['data']['page_metadata']['metatags']
boards_nb = xpath_root['pinterestapp:boards']
pins_nb = xpath_root['pinterestapp:pins']
account_title = xpath_root['title']
description = xpath_root['pinterestapp:about']

print(f"Pinterest account: {account_title}")
print(f"About: {description}")
print(f"Number of boards: {boards_nb}")
print(f"Number of pins: {pins_nb} ")


#%%

# TODO : find all boardnames, only from lindaarden
# find boardnames xpath in json 
board_names = []
board_urls = []
board_pin_counts = []
len_boards = len(json_meta['resourceResponses'][1]['response']['data'])
for i in range(len_boards): 
    board_name = json_meta['resourceResponses'][1]['response']['data'][i]['name']
    board_url = json_meta['resourceResponses'][1]['response']['data'][i]['url']
    board_pin_count = json_meta['resourceResponses'][1]['response']['data'][i]['pin_count']
    board_names.append(board_name)
    board_urls.append(board_url)
    board_pin_counts.append(board_pin_count)

len(board_names)
               
#%%
# single pinterest board url 
URL_String=input("Please enter your Pinterest board url {https://www.pinterest.com/username/board-slug}: ")

username = URL_String.split("/")[3]
boardname = URL_String.split("/")[4]

#%%
#many pinterest boards urls 
URL_String = "https://www.pinterest.com"+board_urls[6]
boardname = board_names[6]

#%%

# download pins from board 

 
import os 
import requests
from bs4 import BeautifulSoup
import json

# set directory
directory= r"D:\thesis\code\ColorCode12\pinterest"
os.chdir(directory)
FOLDER_URL = os.path.join(directory, boardname)
try: 
    os.mkdir(boardname)
    os.chdir(FOLDER_URL)
except:     
    os.chdir(FOLDER_URL)

print("Pins from board " + URL_String + " will be saved to " + FOLDER_URL)

# get correct status code
page=requests.get(URL_String)
print(page.status_code)

# get json from remote pinterest page board 
soup = BeautifulSoup(page.content, 'html5lib')
s = soup.find_all("script", {"id": "initial-state"})
data = soup.select("[type='application/json']")[1]
json1 = json.loads(data.text)



## save json list 
#with open(f"pinterest_{boardname}.json", "w") as write_file:
#    json.dump(json1, write_file)

# find pins xpath in json 
xpath_root = json1['resourceResponses'][0]['response']['data']['images']
xpath = json1['resourceResponses'][0]['response']['data']['images']['474x']
urls = []
dominant_hexcodes = []
for i in range(len(xpath)):
    url = xpath[i]['url']
    urls.append(url)
    hexcode = xpath[i]['dominant_color']
    dominant_hexcodes.append(hexcode)


# TODO: more than only 10 images 
print("Total number of pins " + str(len(urls)))
          
# save pins to PC 
import shutil 
for i in range(len(urls)):
    print("Saving Image Number: " + str(i))
    print("Pin " + f"{urls[i]}" + " processed")
    r = requests.get(urls[i], stream=True)
    with open(f"{i}_"+f"{dominant_hexcodes[i]}_"+urls[i][-34:], 'wb') as f:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        # Copy the response stream raw data to local image file.
        shutil.copyfileobj(r.raw, f)
        # Remove the image url response object.
        del r

