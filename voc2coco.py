import argparse
import xml.etree.ElementTree as ET
import os
import json


#category_item_id = 0
image_id = 20180000000
annotation_id = 0

def get_args_parser():
    parser = argparse.ArgumentParser('Set data dir', add_help=False)
    parser.add_argument('voc_data_dir', type=str)

    return parser
 
def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids
 
#split ='train' 'va' 'trainval' 'test'
def parseXmlFiles_by_txt(data_dir,json_save_path,split='train'):
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    #category_set = dict()
    category_set = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
                'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18,
                'train': 19, 'tvmonitor': 20}
    image_set = set()

    def addCatItem():
        #global category_item_id
        for k, v in category_set.items():
            category_item = dict()
            category_item['supercategory'] = 'none'
            category_item_id = v
            category_item['id'] = category_item_id
            category_item['name'] = k
            coco['categories'].append(category_item)

    def addImgItem(file_name, size):
        global image_id
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        image_id += 1
        image_item = dict()
        image_item['id'] = image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        coco['images'].append(image_item)
        image_set.add(file_name)
        return image_id
 
    def addAnnoItem(object_name, image_id, category_id, bbox):
        global annotation_id
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
 
        annotation_item['segmentation'].append(seg)
 
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        annotation_id += 1
        annotation_item['id'] = annotation_id
        coco['annotations'].append(annotation_item)


    addCatItem()
    labelfile=split+".txt"
    image_sets_file = data_dir + "/ImageSets/Main/"+labelfile
    ids=_read_image_ids(image_sets_file)
 
    for _id in ids:
        xml_file=data_dir + f"/Annotations/{_id}.xml"
 
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
 
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
 
            if elem.tag == 'folder':
                continue
 
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in image_set:
                    raise Exception('file_name duplicated')
 
            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None
 
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    current_category_id = category_set[object_name]
 
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
 
                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)
 
                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                   bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'))
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert VOC2007 annotations to COCO format json file script', parents=[get_args_parser()])
    args = parser.parse_args()

    voc_data_dir = args.voc_data_dir + "/VOC2007"
    json_save_path="./voc_train.json"
    parseXmlFiles_by_txt(voc_data_dir,json_save_path,"trainval")

    image_id = 20180000000
    annotation_id = 0

    json_save_path="./voc_val.json"
    parseXmlFiles_by_txt(voc_data_dir,json_save_path,"test")
