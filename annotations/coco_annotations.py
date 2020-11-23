train_annotations = 'D:/Computer Vision Project/Eurocity Dataset/ECP_day_labels_train/ECP/day/labels/train/amsterdam' \
                    '/amsterdam_00018.json '

import json
import pprint


#
def main():
    json_file = train_annotations
    with open(json_file, 'r') as COCO:
        coco = json.loads(COCO.read())
        # pprint.pprint(coco)
        # ''.join(filter(str.isalpha, input))
        # print(json.dumps(coco['tags'][0]))
        # children is a list not a dictionary have to select one of the items :
        # https://stackoverflow.com/questions/23306653/python-accessing-nested-json-data
        # print(json.dumps(coco['children'][0]['tags']))

        for i in range(len(coco['children'])):
            # print(len(coco['children']))'
            # return
            pprint.pprint(coco['children'][i]['identity'])
            pprint.pprint(coco['children'][i]['x0'])
        '''
         for children in coco['children']:
            print(children)
            print('TEST')
            for key in children:
                print(key)
        '''


if __name__ == "__main__":
    main()
