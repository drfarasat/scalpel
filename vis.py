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

def main():
    # Assuming images are in 'images' folder and json files in 'jsons' folder
    image_folder = '/home/fmr/Downloads/scalpel/images'
    json_folder = '/home/fmr/Downloads/scalpel/jsons'

    for file in os.listdir(image_folder):
        # Construct paths
        img_path = os.path.join(image_folder, file)
        json_name = f'{os.path.splitext(file)[0]}.json'
        json_path = os.path.join(json_folder, json_name)

        # Check if corresponding json exists
        if os.path.exists(json_path):
            draw_bounding_boxes(img_path, json_path)

if __name__ == "__main__":
    main()
