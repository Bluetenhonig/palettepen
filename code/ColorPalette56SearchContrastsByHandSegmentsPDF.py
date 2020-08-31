# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:25:48 2020

@author: Linda Samsinger

=====================
Classification Visualization
=====================

BOTTOM-UP APPROACH:
Use manually-classified segments from movies in the Movie Frame Dataset MVD
as control. 

Caveat: The ERC FilmColors group has only made classifications on segment
level, the machine learning classifier however predicts on screenshot
level. Based on a conference call (7. July 2020), Prof. Pajarola suggested
that the ML classifier's predictions for each screenshot are handed in to
expert Prof. Flückiger for checking the ML classifier's goodness. 

after: ColorPaletteSearchContrastsMLpred23.py 

"""


# import modules  
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import fpdf
from fpdf import FPDF

# USER SPECIFICATION 
IMG_NUMBER = 45447

# declare variables 
DATASET_PATH = r'D:\thesis\film_colors_project\sample-dataset'
SCREENSHOT_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
OUTPUT_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\task3_whole_color_contrasts_annotation'
EXTENSION = '.jpg'

#%%

def load_data(path): 
    os.chdir(DATASET_PATH)
    data = pd.read_excel('dataset.xlsx', index_col=[0])    
    return data


def load_all_images_from_folder(path): 
    images = []
    for r, d, f in os.walk(path):
        for file in f:
            if EXTENSION in file:
                images.append(file) 
    return images 

def get_data(data): 
    movie = data[data.project_id==7]
    contrasts = movie.columns[-11:]
    print(contrasts)
    cols = movie.columns
    return movie, cols 

def get_analysis(data): 
    no_of_segm = data['segment_id'].nunique()  
    no_of_scrn = data['screenshot_id'].nunique() 
    range_of_segm = [data['segment_id'].min(), data['segment_id'].max()]
    range_of_scrn = [data['screenshot_id'].min(), data['screenshot_id'].max()]  
             
    print('Number of segments: ', no_of_segm)
    print('Number of screenshots: ',no_of_scrn)
    return no_of_segm, no_of_scrn, range_of_segm, range_of_scrn

def process_images(data, images): 
    images = sorted(images)
    print('First five images: ', images[:5]) 
    segments= data['segment_id'].unique().tolist()
    print('First five segments: ', segments[:5]) 
    data = data.sort_values(by=['segment_id'], ascending=True)
    data[['screenshot_id', 'segment_id']].head()
    imagespersegment = data['screenshot_id'].groupby(data['segment_id']).agg('count').to_frame()
    return images, data, imagespersegment 


def show_all_contrast_annotated_images(data, images, topn): 
    """" annotations are at segment-level
    warning: annotations displayed here ate image-level are not correct"""
    print('------------------------') 
    print(cols[0], ':', data[cols[0]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[3], ':', data[cols[3]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[4], ':', data[cols[4]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[5], ':', data[cols[5]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[6], ':', data[cols[6]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
       
    for img in images[:topn]: 
        print('------------------------')           
        print(cols[1], ':', data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        print(cols[2], ':', data[cols[2]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        image = cv2.imread(img)           
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image) 
        plt.axis('off')
        plt.show()
        image.shape
        for i in range(-11, 0): 
            print(cols[i], ':', data[cols[i]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])

def show_contrast_annotated_images(data, images, topn): 
    print('------------------------') 
    print(cols[0], ':', data[cols[0]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[3], ':', data[cols[3]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[4], ':', data[cols[4]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[5], ':', data[cols[5]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    print(cols[6], ':', data[cols[6]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0]) 
    before = [0]    
    for img in images[:topn]: 
        print('------------------------')   
        if data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] != before[-1]:         
            print('------------------------')
            print('------NEW SEGMENT-------')
            print('------------------------')
        
        print(str(cols[1]), ':', str(data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0]))
        before.append(data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        print(cols[2], ':', data[cols[2]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        image = cv2.imread(img)           
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image) 
        plt.axis('off')
        plt.show()
        image.shape
        eln = 0 
        for i in range(-11, 0): 
            if data[cols[i]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] == True: 
                print(cols[i])
                eln += 1
        if eln > 2:
            break

def show_contrast_annotation_single_image(img_number): 
    row = data[cols[-11:]][data['screenshot_id'] == 45447]
    print(row.iloc[0])


        
def make_pdf_images_with_contrast_annotation(data, images, path): 
    """ save images to pdf with contrast classification """
    os.chdir(path)
    pdf = fpdf.FPDF(format='A4')
    title = 'FilmColors Project: \nFilm Screenshots and Information about Color Contrasts'
    pdf.set_title(title)
    pdf.set_author('Linda Samsinger') 
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=20)
    pdf.multi_cell(0, 10, title, align = 'L')
    pdf.ln()
    pdf.set_font("Arial", size=18)
    a = f'{cols[3]} : '+ str(data[cols[3]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    b = f'{cols[0]} : ' + str(data[cols[0]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    c = f'{cols[4]} : ' + str(data[cols[4]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    d = f'{cols[5]} : ' + str(data[cols[5]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    e = f'{cols[6]} : ' + str(data[cols[6]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    pdf.multi_cell(0, 10, a, align = 'L')
    pdf.multi_cell(0, 10, b, align = 'L')
    pdf.multi_cell(0, 10, c, align = 'L')
    pdf.multi_cell(0, 10, d, align = 'L')
    pdf.multi_cell(0, 10, e, align = 'L')
    pdf.multi_cell(0, 10, '---------------------------------------', align = 'L')
    pdf.multi_cell(0, 10, 'Number of segments: '+ str(no_of_segm) + ' ' + str(range_of_segm), align = 'L')
    pdf.multi_cell(0, 10, 'Number of screenshots: '+ str(no_of_scrn) + ' ' + str(range_of_scrn), align = 'L')
    pdf.set_font("Arial", size=12)
    i = 0  
    before = [0]
    for idx, img in enumerate(images):
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 20, "Screenshot: " + str(idx+1))
        pdf.set_font("Arial", size=12)
        str1 = f'{cols[1]} : ' + str(data[cols[1]][data["screenshot_id"] == np.array(img[:-4], dtype="uint64")].iloc[0])
        pdf.multi_cell(0, 5,str1)
        pdf.ln()
        str2 = f'{cols[2]} : ' + str(data[cols[2]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        pdf.multi_cell(0, 0,str2)
        if data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] != before[-1]:         
            pdf.text(165, 20, 'Start of Segment')
        before.append(data[cols[1]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
        pdf.image(img, x = 10, y = 50, w = 50, h = 50, type = 'JPG') 
        pdf.multi_cell(0, 135, 'Color Contrasts: ')
        eln = 0 
        for i in range(-11, 0):      
            if data[cols[i]][data['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] == True: 
                el = cols[i] 
                if eln == 0 : 
                    pdf.multi_cell(0, -115,el)
                eln += 1
                if eln == 2: 
                    pdf.multi_cell(0, 130, el)
                if eln == 3: 
                    pdf.multi_cell(0, -115, el)
    pdf.output("screenshots_contrasts.pdf")



class PDF(FPDF):
    # add page numbers 
    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')
    
   
def make_multipage_pdf_images_with_contrast_annotation(data, images, path):  
    """ save images to multiple-pages pdf with contrast classification"""   
    os.chdir(path)
    pdf = PDF(format='A4')
    title = 'FilmColors Project: The Movie Frame Dataset \nFilm Screenshots and their Classification into Color Contrasts'
    pdf.set_title(title)
    pdf.set_author('Linda Samsinger')
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=20)
    pdf.multi_cell(0, 10, title, align = 'L')
    pdf.ln()
    pdf.set_font("Arial", size=18)
    a = f'{cols[3]} : '+ str(data[cols[3]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    b = f'{cols[0]} : ' + str(data[cols[0]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    c = f'{cols[4]} : ' + str(data[cols[4]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    d = f'{cols[5]} : ' + str(data[cols[5]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    e = f'{cols[6]} : ' + str(data[cols[6]][data['screenshot_id'] == np.array(images[0][:-4], dtype='uint64')].iloc[0])
    pdf.multi_cell(0, 10, a, align = 'L')
    pdf.multi_cell(0, 10, b, align = 'L')
    pdf.multi_cell(0, 10, c, align = 'L')
    pdf.multi_cell(0, 10, d, align = 'L')
    pdf.multi_cell(0, 10, e, align = 'L')
    pdf.multi_cell(0, 10, '---------------------------------------', align = 'L')
    pdf.multi_cell(0, 10, 'Number of segments: '+ str(no_of_segm) + ', segments = ' + str(range_of_segm), align = 'L')
    pdf.multi_cell(0, 10, 'Number of screenshots: '+ str(no_of_scrn) + ', screenshots = ' + str(range_of_scrn), align = 'L')
    pdf.set_font("Arial", size=12)
    i = 0  
    print('Total: ', len(SEGMENTS))
    for idx, seg in enumerate(SEGMENTS):
        print(idx, ':', seg, 'processed.')
        pdf.add_page()
        range_scrns = [data['screenshot_id'][data['segment_id'] == seg].min(), data['screenshot_id'][data['segment_id'] == seg].max()] 
        str1 = f'{cols[1]} : ' + str(seg)
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 20, "Segment: " + str(idx+1))
        pdf.set_font("Arial", size=12)
        pdf.text(36,21, ' (' + str1 + ')')    
        pdf.multi_cell(0, 0, "Number of screenshots: " + str(groups.loc[seg].iloc[0]) + ', screenshots = ' + str(range_scrns))   
        pdf.multi_cell(0, 20, 'Color Contrasts: ')
        eln = 0
        total = []
        for i in range(-11, 0):      
            if data[cols[i]][data['segment_id'] == seg].iloc[0] == True: 
                el = cols[i]
                el = el.replace('Color Contrasts:', '- ')
                eln += 1
                if eln == 1: 
                    pdf.multi_cell(0, 0,el)            
                if eln == 2: 
                    pdf.multi_cell(0, 10, el)
                if eln == 3: 
                    pdf.multi_cell(0, 0, el)
                total.append(el)
        if total == []: 
            pdf.multi_cell(0, 0, 'None')
        seg_images = [str(img) + '.jpg' for img in data['screenshot_id'][data['segment_id'] == seg]]
        seg_images = sorted(seg_images)
        seg_img_len = len(seg_images)
        sequence = np.arange(0, seg_img_len, 3).tolist()
        sequence.append(sequence[-1]+3)
        break_seq = np.arange(-1, seg_img_len, 9).tolist()
    
        y= 0
        for ids in range(len(sequence)-1):
            x= 0
            y+=70
            for tri in range(sequence[ids], sequence[ids+1]):               
                if sequence[ids] %6 == 0 and tri < seg_img_len:                 
                    #print(tri)
                    pdf.text(10+x, 68+y-70, seg_images[tri])
                    try: 
                        pdf.image(seg_images[tri], 10+x, y, w = 50, h = 50, type = 'JPG')
                    except: 
                        pass
                    x += 60
                    
                if sequence[ids] %6 == 3 and tri < seg_img_len: 
                    pdf.text(10+x, 68+y-70, seg_images[tri])
                    try: 
                        pdf.image(seg_images[tri], 10+x, y, w = 50, h = 50, type = 'JPG')
                    except: 
                        pass
                    x += 60
                if tri in break_seq and tri !=0 and tri != seg_img_len-1: 
                    pdf.add_page()
                    y=0       
    pdf.output("screenshots_contrasts_multi_all.pdf")   
    
#%%

if __name__ == '__main__': 
    
    data = load_data(DATASET_PATH)
    data, cols = get_data(data)

    os.chdir(SCREENSHOT_PATH)
    images = load_all_images_from_folder(SCREENSHOT_PATH)
    
    no_of_segm, no_of_scrn, range_of_segm, range_of_scrn = get_analysis(data)

    images, data, imagespersegment  = process_images(data, images)

#%%
       
    # contrast annotation
    
    # single images 
    show_contrast_annotation_single_image(IMG_NUMBER)
    
    # all images
    show_all_contrast_annotated_images(data, images, 5)
    show_contrast_annotated_images(data, images, 5)


#%%

    # make pdf with contrast annotation: single and multi-pages
    make_pdf_images_with_contrast_annotation(data, images, OUTPUT_PATH)
    make_multipage_pdf_images_with_contrast_annotation(data, images, OUTPUT_PATH)
    


 