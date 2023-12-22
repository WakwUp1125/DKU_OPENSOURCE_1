오픈소스AI응용[텀프로젝트]
object detection을 이용한 아동 심리 분석
모형설명
retina net
- One Stage Detector의 빠른 detection 시간의 장점을 가지면서 One stage detector의 detection 성능 저하 문제 개선
- 수행 시간은 YOLO나 SSD보다 느리지만 Faster RCNN보다 빠름
- 수행 성능은 타 detection 모델보다 뛰어남. 특히 One stage detector 보다 작은 object에 대한 detection능력이 뛰어남

retina net은 하나의 Backbone Network와 각각의 Classificationrhk Bounding Box Regression을 수행하는 2개의 Subnetwork로 구성
<img width="80%" src="https://github.com/WakwUp1125/DKU_OPENSOURCE_1/assets/130390077/af3001cd-6316-4ea4-ba3a-80faf2277329"/>
