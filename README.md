# train.py 실행 명령어(코드 테스트)

```bash
python train.py --feature_dim 12 --real real_temp --fake fake_temp --batch_size 2 --epochs 4 --model lstm --learning_rate 0.00001 --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4'
```

# train.py 실행 명령어(실제 학습)

```bash
python train.py --feature_dim 12 --real LJSpeech-1.1/wavs --fake generated_audio/ljspeech_hifiGAN --batch_size 32 --epochs 100 --model lstm --learning_rate 0.00001 --mfcc_feature_idx 'all' --evs_feature_idx 'none' --lsf_feature_idx 'none' --lsp_feature_idx 'none'
```



# 명령인수 설명
- feature_dim : 뽑아내는 feature와 mfcc 계수의 개수(차원)
- real : real 데이터 있는 폴더(이 안에 wav, feature_{feature_dim} 폴더 있음)
- fake : real 데이터 있는 폴더(이 안에 wav, feature_{feature_dim} 폴더 있음)
    - 코드 테스트 시에는 --real, --fake에는 real_temp, fake_temp로 설정하고, 실제 학습 시에는 --real LJSpeech-1.1/wavs --fake generated_audio/ljspeech_hifiGAN으로 설정
- model : 현재는 lstm, cnn중에 선택(우선 lstm만 가능)
- mfcc_feature_idx : 뽑아내는 행 번호(mfcc)
- evs_feature_idx : 뽑아내는 행 번호(evs)
    - 'all'의 경우 전부 다
    - 'none'의 경우 하나도 안뽑음

- lsf_feature_idx : 파이썬으로 lsf를 feature_dim만큼 추출해서 사용
- lsp_feature_idx : 파이썬으로 lsp를 feature_dim만큼 추출해서 사용


# 데이터셋 준비
- 여러 종류 데이터셋 사용 가능하지만 LJSpeech를 기본으로 함
- real 데이터셋 : https://keithito.com/LJ-Speech-Dataset/
- fake 데이터셋 : https://zenodo.org/record/5642694
    - 해당 음성파일은 22050Hz로 샘플링 되어있음(코드 내에 16kHz로 리샘플링 하는 부분 포함되어있음)
- evs 코덱을 통해 추출한 feature 파일 : 
    - evs feature로 학습시키려면 features_{feature_dim} 폴더가 현재 작업폴더 내에 있어야 한다
    - 해당 폴더가 없을 경우, features_12에서 feautre를 가져오고 제로패딩을 하여 진행하므로, 데이터 링크는 features_12만 올려놓았다

- 해당 데이터셋을 모두 다운받은 뒤, 압축해제한 폴더를 현재 작업폴더(AudioDeepFakeDetection 폴더) 하위에 위치시킨다


# 진행상황 및 주의사항(수시로 업데이트됨)
- (2024/08/02) evs에서 모든 서브프레임에 대한 lsp 추출하였음
    - 파이썬으로 lsp 추출하는 방법 검토해볼 필요 있음
- (2024/07/03) run_lsp.sh, run_lsf.sh, run_mfcc.sh 실행하면 각각 mfcc '20 ~ 31'에 lsp, lsf, 저차원 mfcc 쌓아서 학습 진행
    - 총 10번의 학습을 진행하고 성능지표 평균 및 최대최소 알려줌
    - run.sh 실행 시 새 쉘 스크립트 모두 실행됨
- (2024/07/03) feature_{feature_dim} 폴더 없을때 feautre_12에서 데이터 뽑고 zero padding 하게 수정함
    - 또한, 데이터 pt 파일도 'lj_hifi_{feature_dim}' 폴더에 저장하고 여기서 불러오게 만들었다
- (2024/07/03) 조기종료, dropout 적용됨 
- (2024/07/02) mfcc는 pt파일로 저장하지 말고 그냥 음원에서 뽑자
    - 40차 이상이 넘어가니까 죽는다. 
- (2024/07/01) 현재 서버의 pt 파일들은 모두 정규화된 12차 데이터이다
- 현재 코드에서 mfcc_feature_idx를 'all'로 두면 고정된 filter back 수인 50이 모두 나옴
    - 우선은 수정하지 않음(추후에 수정 방향 생각)
- (2024/06/30) Dataset.py에서, feature_dim이 클 경우 pt파일로 저장하는 부분 주석처리 해야함
- model을 cnn으로 설정했을 경우, 'mfcc_feature_idx의 크기 + evs_feature_idx의 크기'가 8 이상이어야 함
- 예를 들어 '--model cnn --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4' 로 설정하면 실행 안됨
- lstm은 상관 없음

- 현재 AudioDeepFakeDetectionNew 리포지토리에는 데이터가 있는 폴더는 있지 않다

- 현재 작업 폴더에 features_labels_mfcc_{feature_dim}.pt, features_labels_evs_{feature_dim}.pt 있으면 미리 저장된 데이터 불러온다
- 서버(선배님 컴퓨터)에서 현재 폴더의 pt 파일은 real은 LJ, fake는 LJ melgan 저장되어있다
- 학습할 때마다 features_and_labels 폴더에 학습 데이터를 pt 파일로 저장한다(추후 학습을 위해)
- 단, 명령 인수와 데이터 형태가 일치하는지 잘 확인해야한다!!

# 주의사항
- model을 cnn으로 설정했을 경우, 'mfcc_feature_idx의 크기 + evs_feature_idx의 크기'가 8 이상이어야 함
- 예를 들어 '--model cnn --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4'' 로 설정하면 실행 안됨
- lstm은 상관 없음

- 현재 AudioDeepFakeDetectionNew 리포지토리에는 데이터가 있는 폴더는 있지 않다

# inference.py 실행 명렁어
```bash
 python inference.py --model model_weights.pt --model_architecture lstm --mfcc_feature_idx '0 1 3' --evs_feature_idx '2 4' --feature_dim 12 --input_dir fake
```


# git 사용법
0. 새로운 리포지토리와 연결하려면, 현재 작업폴더의 .git 폴더를 삭제하고, 초기화한다

```bash
rm -rf .git # .git 폴더 삭제
git init # 초기화
```
1. 원격 저장소 연결
```bash
git remote add <원격 저장소 주소>
```
- 원격 저장소 확인
```bash
git remote -v
```

2. 브랜치 확인
```bash
git branch
```
- 보통 master 또는 main

3. 업로드할 파일 추가
```bash
git add *.py
```

4. 커밋
```bash
git commit -m "initial commit"
```

5. 푸쉬
```bash
git push origin master
```
또는
```bash
git push origin main
```
- 푸쉬할 때 계정명과 비밀번호를 물어볼 수 있음
- 계정명은 TTooBee, 비밀번호는 서버 바탕화면에 '깃허브 비밀번호(토큰)' 파일에 적혀있음