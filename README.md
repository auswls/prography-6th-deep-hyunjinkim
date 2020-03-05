# prography-6th-deep-hyunjinkim
프로그라피 사전 과제를 위한 Repository 입니다



## 기본 정보
* 프레임 워크 : pytorch
* training code : ```python train.py```
* test code : ```python test.py```

  + ```test.py``` 실행시 모델이 위치한 경로에 따라 ```torch.load()```의 path를 변경해주세요!
* sample image에 대한 prediction & image 출력 코드 : ```python inference.py```

  + ```inference.py``` 실행시 sample image 폴더에 대한 path를 변경해주세요!
* 모델 저장 위치 : 제 Google Drive(https://drive.google.com/open?id=1G8Cf_kYvza3uNutX4R5ftnIcAMGewAv0) 에 저장해두었습니다.

  + 필수사항 5번의 accuracy를 참고하여 적합한 모델(accuracy가 가장 높은 모델)을 사용해주세요.


## 필수사항
> #### 1. VGG-16으로 네트워크를 구성하고, MNIST 데이터를 RGB 채널로 변경해주세요.
> * VGG-16 네트워크 구성은 model.py 파일에 구현해두었습니다.
> * RGB 채널로의 변경은 rgb_transform 함수로 구현해두었습니다.


> #### 2. (1)의 모델 구조에서 model initialization, inference 부분을 함수형태로 작성해주세요.
> * initialization 부분은 model.py에 구현해 두었고, inference 부분은 inference.py 파일로 분리해두었습니다.
> * inference.py에서는 sample에 있는 sample images를 받아 prediction한 후 결과를 출력해줍니다.


> #### 3. (2)의 구조에서 Conv2_1의 입력을 첫번째 Dense 입력에 추가해주는 구조를 추가해주세요. (Skip connection 구조)
> * model.py에 self.skip를 구현, conv2_1의 input을 ```self.skip```에 넣어 input size를 조절한 후, ```torch.cat```을 이용해 두 input을 결합합니다.


> #### 4. (3)에서 나온 모델을 RGB채널로 바꾼 MNIST로 학습해주세요.
> * train.py를 실행시켜 학습시켜 주었고, train.py 내의 rgb_transform 함수를 이용해 MNIST를 RGB채널로 바꿔주었습니다.
> * epoch가 끝날 때마다 accuracy와 average loss가 출력되고, pth 형식의 파일로 모델이 저장됩니다.

> #### 5. ```python test.py``` 을 통해 테스트 코드를 실행시켜 정확도를 출력해 주세요.
> * 실행 결과 나온 Accuracy와 Average loss 입니다.
> ```
> Epoch 1 : Test set: Average loss: 0.0506, Accuracy: 9366/10000 (94%)
> Epoch 2 : Test set: Average loss: 0.0389, Accuracy: 9520/10000 (95%)
> Epoch 3 : Test set: Average loss: 0.0399, Accuracy: 9513/10000 (95%)
> Epoch 4 : Test set: Average loss: 0.0304, Accuracy: 9617/10000 (96%)
> Epoch 5 : Test set: Average loss: 0.0356, Accuracy: 9553/10000 (96%)
> Epoch 6 : Test set: Average loss: 0.0335, Accuracy: 9616/10000 (96%)
> Epoch 7 : Test set: Average loss: 0.0332, Accuracy: 9620/10000 (96%)
> Epoch 8 : Test set: Average loss: 0.0377, Accuracy: 9565/10000 (96%)
> Epoch 9 : Test set: Average loss: 0.0341, Accuracy: 9620/10000 (96%)
> Epoch 10 : Test set: Average loss: 0.0293, Accuracy: 9652/10000 (97%)
> ```

> #### 6. 정확도와 구현한 모델의 ADT를 README.md에 간단히 요약해주세요.
> * 기본 VGG-16 모델에서 Conv2_1의 input을 첫번째 Dense input에 추가하기 위해서 [1, 64, 112, 112]인 사이즈를 [1, 512, 7, 7]로 바꿔줍니다
> * 이후 ```torch.cat```을 이용해 Conv2_1의 input을 첫번째 Dense input에 추가해준 후 첫번째 Dense layer에 합쳐진 input이 들어가도록 합니다.
> ```
> self.skip = nn.Conv2d(64, 512, kernel_size = 3, padding = 1, stride = 18)
> x = torch.cat((x, skip_connection), dim = 1)
> ``` 
