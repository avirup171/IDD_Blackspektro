import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path,filepath):
    xml_list = []
    xml_file=path
    print(xml_file)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (filepath,
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[1][0].text),
                int(member[1][1].text),
                int(member[1][2].text),
                int(member[1][3].text)
                )
        xml_list.append(value)
    column_name = ['filepath','filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list)
    return xml_df


def main():
    column_name = ['filepath','filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(columns=column_name)
    xml_df.to_csv(('images/test_labels.csv'),mode='a', index=None)
    '''
    with open('train.txt') as t:
        for folder in t:
            filepath="JPEGImages/"+folder.rstrip()+'.jpg'
            folder='Annotations/'+folder.rstrip()+'.xml'
            #print(folder)
            image_path = os.path.join(os.getcwd(), (folder))
            #print(image_path)
            xml_df = xml_to_csv(image_path,filepath)
            xml_df.to_csv(('images/train_labels.csv'),mode='a', header=False, index=None)
            print('Successfully converted train xml to csv.')
    '''
    with open('test.txt') as t:
        for folder in t:
            filepath="JPEGImages/"+folder.rstrip()+'.jpg'
            folder='Annotations/'+folder.rstrip()+'.xml'
            #print(folder)
            image_path = os.path.join(os.getcwd(), (folder))
            #print(image_path)
            xml_df = xml_to_csv(image_path,filepath)
            xml_df.to_csv(('images/test_labels.csv'),mode='a', header=False, index=None)
            print('Successfully converted test xml to csv.')
   
main()