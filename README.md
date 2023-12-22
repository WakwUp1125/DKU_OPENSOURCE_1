# 오픈소스AI응용[텀프로젝트]

## object detection을 이용한 아동 심리 분석

- 데이터셋을 생성하여  모델을 학습시켜 이미지 캡셔닝을 이용한 아동 심리 분석 시도(오픈소스연습.ipynb,오픈소스_연습모델.ipynb,Untitled.ipynb)
- 이미지 인식에 대한 오류 발생, 라벨링한 데이터 활용에 대한 어려움 -> 이미지 탐지 모델 변경
- Retina Net을 활용한 object detection 결과에 따른 우울 정도 계산(object_detection2.ipynb)
  
### 모형설명

#### Retina Net
- One Stage Detector의 빠른 detection 시간의 장점을 가지면서 One stage detector의 detection 성능 저하 문제 개선
- 수행 시간은 YOLO나 SSD보다 느리지만 Faster RCNN보다 빠름
- 수행 성능은 타 detection 모델보다 뛰어남. 특히 One stage detector 보다 작은 object에 대한 detection능력이 뛰어남


Retina Net은 하나의 Backbone Network와 각각의 Classification과 Bounding Box Regression을 수행하는 2개의 Subnetwork로 구성

<img width="80%" src="https://github.com/WakwUp1125/DKU_OPENSOURCE_1/assets/130390077/af3001cd-6316-4ea4-ba3a-80faf2277329"/>

#### 결과
<img width="240" src="https://github.com/WakwUp1125/DKU_OPENSOURCE_1/assets/130390077/de4a55dc-3f20-4cd5-ac0e-15ccdbda4349"/>
<img width="240" src="https://github.com/WakwUp1125/DKU_OPENSOURCE_1/assets/130390077/30b734d6-ecd6-4219-a087-b2d815645d6c"/>
<img width="240" src="https://github.com/WakwUp1125/DKU_OPENSOURCE_1/assets/130390077/00cee1bd-0f4a-4876-9c18-30d828336ad1"/>


### 요구 사항

python 3.10.12

tensorflow 2.15.0

### 실행방법(예시 : 사람 그림 분석)

#### 1. import libraries
~~~
import os
import requests
from PIL import Image
from io import BytesIO
import time
import tensorflow as tf
~~~

#### 2. 객체 탐지
~~~
def download_and_resize_image(url, width, height):
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to download image. HTTP status code: {response.status_code}")
            return None

        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")

        temp_dir = "/path/to/temp/"
        os.makedirs(temp_dir, exist_ok=True)

        image = image.resize((width, height), Image.LANCZOS)

        temp_file_path = os.path.join(temp_dir, "temp_image.jpg")
        image.save(temp_file_path)

        return temp_file_path
    except Exception as e:
        print(f"Error downloading or resizing image: {e}")
        return None

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, img):
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    return result

def detect_and_output_objects(image_url):
    image_path = download_and_resize_image(image_url, 640, 480)

    if image_path is not None:
        start_time = time.time()
        img = load_img(image_path)
        objects = run_detector(detector, img)

        if objects is not None:
            end_time = time.time()
            print("Inference time:", end_time - start_time)

            for i in range(len(objects["detection_scores"])):
                score = objects["detection_scores"][i]
                class_name = objects["detection_class_entities"][i].decode("utf-8") 
                box = objects["detection_boxes"][i]
                ymin, xmin, ymax, xmax = box.tolist()

                im_height, im_width, _ = img.shape
                ymin, xmin, ymax, xmax = ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width

                print(f"Object: {class_name}")
                print(f"  - Score: {score}")
                print(f"  - Box: ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}")
                print()

        else:
            print("Object detection failed.")
    else:
        print("Image download or processing failed.")

image_urls = ["https://drive.google.com/uc?id=1T1Su-UhZRQ-krH05Y0pc_d855eSRafnB"]
detect_and_output_objects(image_urls[0])
~~~
#### 3. 객체 추출
~~~
def extract_desired_objects(objects, img):
    desired_objects = ["Person", "Human head", "Human eye", "Human nose", "Human neck","Human mouth", "Human body", "Human arm",
                       "Human leg", "Human hand", "Foot"]

    object_dict = {}

    for i in range(len(objects["detection_scores"])):
        score = objects["detection_scores"][i]
        class_name = objects["detection_class_entities"][i].decode("utf-8")
        box = objects["detection_boxes"][i]
        ymin, xmin, ymax, xmax = box.tolist()

        im_height, im_width, _ = img.shape
        ymin, xmin, ymax, xmax = ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width

        if class_name in desired_objects:
            if class_name not in object_dict or score > object_dict[class_name]["Score"]:
                object_dict[class_name] = {
                    "Score": score,
                    "Box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
                }

    extracted_objects = [{"Object": key, **value} for key, value in object_dict.items()]

    if not extracted_objects:
        return None

    return extracted_objects

def detect_and_output_desired_objects(image_url):
    image_path = download_and_resize_image(image_url, 640, 480)

    if image_path is not None:
        start_time = time.time()
        img = load_img(image_path)
        objects = run_detector(detector, img)

        if objects is not None:
            end_time = time.time()
            print("Inference time:", end_time - start_time)

            desired_objects = extract_desired_objects(objects, img)

            if desired_objects is not None:
                for obj in desired_objects:
                    print(f"Object: {obj['Object']}")
                    print(f"  - Score: {obj['Score']}")
                    print(f"  - Box: ymin={obj['Box']['ymin']}, xmin={obj['Box']['xmin']},
                          ymax={obj['Box']['ymax']}, xmax={obj['Box']['xmax']}")
                    print()
            else:
                print("No desired objects detected.")
        else:
            print("Object detection failed.")
    else:
        print("Image download or processing failed.")

image_urls = ["https://drive.google.com/uc?id=1T1Su-UhZRQ-krH05Y0pc_d855eSRafnB"]
detect_and_output_desired_objects(image_urls[0])
~~~
#### 4. 우울 점수 계산
~~~   
def calculate_depression_score(obj_type, ymin, ymax, image_height):
    default_depression_score = 38.00

    if obj_type == "Person":
        if ymin < image_height / 3:
            return 25.94
        elif image_height / 3 < ymin < 2 * image_height / 3:
            return 26.83
        elif ymax > 2 * image_height / 3:
            return 26.25
        else:
            return default_depression_score
    elif obj_type == "Human head":
        return 26.38
    elif obj_type == "Human eye":
        return 26.32
    elif obj_type == "Human nose":
        return 26.00
    elif obj_type == "Human neck":
        return 26.53
    elif obj_type == "Human mouth":
        return 26.15
    elif obj_type == "Human body":
        return 26.36
    elif obj_type == "Human arm":
        return 26.37
    elif obj_type == "Human leg":
        return 26.43
    elif obj_type == "Human hand":
        return 26.18
    elif obj_type == "Foot":
        return 26.18
    else:
        return default_depression_score

def extract_desired_objects(objects, img, image_height):
    desired_objects = ["Person", "Human head", "Human eye", "Human nose", "Human neck","Human mouth", "Human body", "Human arm",
                      "Human leg", "Human hand", "Foot"]

    object_dict = {}
    total_score = 0

    for i in range(len(objects["detection_scores"])):
        score = objects["detection_scores"][i]
        class_name = objects["detection_class_entities"][i].decode("utf-8")
        box = objects["detection_boxes"][i]
        ymin, xmin, ymax, xmax = box.tolist()

        im_height, im_width, _ = img.shape
        ymin, xmin, ymax, xmax = ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width

        if class_name in desired_objects:
            if class_name not in object_dict or score > object_dict[class_name]["Score"]:
                depression_score = calculate_depression_score(class_name, ymin, ymax, image_height)
                object_dict[class_name] = {
                    "Score": depression_score,
                    "Box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
                }
                total_score += depression_score

    extracted_objects = [{"Object": key, **value} for key, value in object_dict.items()]

    if not extracted_objects:
        return None, None

    average_score = total_score / len(extracted_objects)

    return extracted_objects, average_score

def detect_and_output_desired_objects(image_url):
    image_path = download_and_resize_image(image_url, 640, 480)

    if image_path is not None:
        start_time = time.time()
        img = load_img(image_path)
        objects = run_detector(detector, img)

        if objects is not None:
            end_time = time.time()
            print("Inference time:", end_time - start_time)

            desired_objects, average_score = extract_desired_objects(objects, img, img.shape[0])

            if desired_objects is not None:
                for obj in desired_objects:
                    print(f"Object: {obj['Object']}")
                    print(f"  - Score: {obj['Score']}")
                    print(f"  - Box: ymin={obj['Box']['ymin']}, xmin={obj['Box']['xmin']},
                           ymax={obj['Box']['ymax']}, xmax={obj['Box']['xmax']}")
                    print()

                print(f"Total Depression Score: {average_score * len(desired_objects)}")
                print(f"Average Depression Score: {average_score}")
            else:
                print("No desired objects detected.")
        else:
            print("Object detection failed.")
    else:
        print("Image download or processing failed.")

image_urls = ["https://drive.google.com/uc?id=1T1Su-UhZRQ-krH05Y0pc_d855eSRafnB"]
detect_and_output_desired_objects(image_urls[0])
~~~
