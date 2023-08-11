import cv2
import os
import json

def draw_bounding_boxes(image_path, json_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract bounding boxes and draw them on the image
    for instrument in data['instruments']:
        bbox = instrument['bbox']
        #x1, y1, x2, y2 = bbox
        a, b, w, h = bbox
        x1 = a
        x2 = x1+w
        y1 = b
        y2 = y1+h

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, instrument['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rescale_dataset(image_path, json_path, newScale=512):
    # we need to rescale the images to 224x224
    # and also the bounding boxes
    # calculate the xscale and yscale
    # xscale = 224 / width
    # yscale = 224 / height
    # new_x1 = x1 * xscale
    # new_x2 = x2 * xscale
    # new_y1 = y1 * yscale
    # new_y2 = y2 * yscale
    # new_width = new_x2 - new_x1
    # new_height = new_y2 - new_y1
    # new_bbox = [new_x1, new_y1, new_width, new_height]
    # new_image = cv2.resize(image, (224, 224))
    # save the new image and json file
    # new_image_path = os.path.join(new_image_folder, file)
    # new_json_path = os.path.join(new_json_folder, json_name)
    # cv2.imwrite(new_image_path, new_image)
    # with open(new_json_path, 'w') as f:
    #     json.dump(new_data, f)
    # return new_image_path, new_json_path
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]


    # calculate the xscale and yscale
    xscale = newScale / width
    yscale = newScale / height
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    new_data = []
    for instrument in data['instruments']:
        bbox = instrument['bbox']
        x1, y1, x2, y2 = bbox
        #a, b, w, h = bbox
        # scale the bbox
        new_x1 = int(x1 * xscale)
        new_x2 = int(x2 * xscale)
        new_y1 = int(y1 * yscale)
        new_y2 = int(y2 * yscale)

        bbox = [new_x1, new_y1, new_x2, new_y2]
        instrument['bbox'] = bbox
        new_data.append(instrument)

    json_data= data
    json_data['instruments'] = new_data
    

    # resize the image
    new_image = cv2.resize(img, (newScale, newScale), interpolation=cv2.INTER_AREA)
    new_image_folder = '/home/fmr/Downloads/scalpel/rescale/images'
    new_json_folder = '/home/fmr/Downloads/scalpel/rescale/jsons'

    # save the new image and json file
    #save the new image and json file
    # 
    file = os.path.basename(image_path)
    json_name = f'{os.path.splitext(file)[0]}.json'

    new_image_path = os.path.join(new_image_folder, file)
    new_json_path = os.path.join(new_json_folder, json_name)
    cv2.imwrite(new_image_path, new_image)
    with open(new_json_path, 'w') as f:
        json.dump(json_data, f)
    return new_image_path, new_json_path
    
    


def main():
    # Assuming images are in 'images' folder and json files in 'jsons' folder
    image_folder = '/home/fmr/Downloads/scalpel/images'
    json_folder = '/home/fmr/Downloads/scalpel/jsons'

    #image_folder = '/home/fmr/Downloads/scalpel/rescale/images'
    #json_folder = '/home/fmr/Downloads/scalpel/rescale/jsons'

    for file in os.listdir(image_folder):
        # Construct paths
        img_path = os.path.join(image_folder, file)
        json_name = f'{os.path.splitext(file)[0]}.json'
        json_path = os.path.join(json_folder, json_name)

        # Check if corresponding json exists
        if os.path.exists(json_path):
            #draw_bounding_boxes(img_path, json_path)
            rescale_dataset(img_path, json_path)

if __name__ == "__main__":
    main()
