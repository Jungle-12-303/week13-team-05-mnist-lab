# 로컬 개발 환경 기본 세팅

> 기준 프로젝트 루트: `week13-team-05-mnist-lab`
>
> 기준 Python: 3.11 계열
>
> 기준 Conda 환경 이름: `mnist-nn`
>
> 모든 명령은 별도 안내가 없으면 프로젝트 루트에서 실행한다.

---

## 1. Conda 설치

로컬에서는 README 기준에 맞춰 Conda 환경을 사용한다. Conda를 쓰면 `.venv`를 따로 만들 필요가 없다.

### Windows

Windows에서는 Anaconda 또는 Miniconda를 설치한다.

- Anaconda: 패키지가 많이 포함된 배포판
- Miniconda: 필요한 패키지만 직접 설치하는 가벼운 배포판

설치 후 PowerShell을 새로 열고 아래 명령이 동작하는지 확인한다.

```powershell
conda --version
```

### Mac

Mac에서는 README 기준에 따라 Miniforge 사용을 권장한다.

설치 후 터미널을 새로 열고 아래 명령이 동작하는지 확인한다.

```bash
conda --version
```

---

## 2. Conda 명령이 인식되지 않을 때

아래 오류가 나오면 Conda가 현재 쉘에서 인식되지 않는 상태다.

```powershell
conda: The term 'conda' is not recognized
```

이 경우 Conda 실행 파일의 전체 경로로 먼저 동작 여부를 확인한다. 설치 위치는 팀원마다 다르므로 본인 PC의 경로로 바꾼다.

Windows PowerShell 예시:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" --version
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" --version
```

일반 형태:

```powershell
& "<conda.exe_경로>" --version
```

Conda 실행 파일이 확인되면 PowerShell 초기화를 실행한다.

```powershell
& "<conda.exe_경로>" init powershell
```

예시:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

그 다음 PowerShell을 완전히 닫고 다시 연다. 다시 연 뒤 아래 명령이 동작하면 설정이 완료된 것이다.

```powershell
conda --version
```

---

## 3. 프로젝트 루트로 이동

저장소를 받은 위치는 팀원마다 다를 수 있으므로, 부모 폴더에서 프로젝트 폴더명으로 이동한다.

```powershell
cd week13-team-05-mnist-lab
```

현재 위치가 프로젝트 루트인지 확인하려면 아래 파일들이 보여야 한다.

```powershell
dir
```

확인할 파일:

- `README.md`
- `requirements.txt`
- `src`
- `tests`

---

## 4. Conda 환경 생성

README 기준에 맞춰 Python 3.11 환경을 만든다.

```powershell
conda create -n mnist-nn python=3.11 -y
```

이미 환경이 만들어져 있다면 다시 만들 필요가 없다. 환경 목록은 아래 명령으로 확인한다.

```powershell
conda env list
```

`mnist-nn`이 목록에 있으면 다음 단계로 넘어간다.

---

## 5. Conda 환경 활성화

```powershell
conda activate mnist-nn
```

정상적으로 활성화되면 프롬프트 앞에 `(mnist-nn)`이 표시된다.

```powershell
(mnist-nn) PS ...
```

Python 버전을 확인한다.

```powershell
python --version
```

`Python 3.11.x`가 나오면 이 프로젝트 기준에 맞다.

---

## 6. 의존성 설치

프로젝트 루트에서 실행한다.

```powershell
python -m pip install -r requirements.txt
```

설치된 패키지를 확인하려면 아래 명령을 사용한다.

```powershell
python -m pip list
```

이 프로젝트의 주요 의존성은 다음과 같다.

- `numpy`
- `matplotlib`
- `pytest`

---

## 7. 기본 테스트 실행

처음 세팅이 끝나면 가장 작은 테스트부터 실행한다.

```powershell
python -m pytest tests/test_relu.py -q
```

아직 구현 전이라면 실패할 수 있다. 이 경우 환경 문제가 아니라 구현 대상 함수가 아직 `NotImplementedError` 상태일 가능성이 높다.

전체 테스트는 단계별 구현이 어느 정도 진행된 뒤 실행한다.

```powershell
python -m pytest tests/ -v
```

---

## 8. `conda activate` 없이 실행하는 방법

PowerShell 초기화가 아직 안 되었거나, 현재 쉘에서 `conda activate`가 불편하면 `conda run`을 사용할 수 있다.

의존성 설치:

```powershell
conda run -n mnist-nn python -m pip install -r requirements.txt
```

Python 버전 확인:

```powershell
conda run -n mnist-nn python --version
```

테스트 실행:

```powershell
conda run -n mnist-nn python -m pytest tests/test_relu.py -q
```

`conda` 명령 자체가 인식되지 않는 경우에는 `conda` 부분을 전체 경로로 바꾼다.

```powershell
& "<conda.exe_경로>" run -n mnist-nn python -m pytest tests/test_relu.py -q
```

---

## 9. 권장 테스트 순서

처음부터 전체 테스트만 실행하지 않는다. 현재 구현 중인 파일의 테스트부터 통과시킨다.

### 1단계: Activation

```powershell
python -m pytest tests/test_relu.py -q
python -m pytest tests/test_softmax.py -q
```

### 2단계: Layer / Loss

```powershell
python -m pytest tests/test_affine.py -q
python -m pytest tests/test_cross_entropy_loss.py -q
```

### 3단계: Optimizer

```powershell
python -m pytest tests/test_sgd.py -q
python -m pytest tests/test_adam.py -q
```

### 4단계: BatchNorm / Dropout

```powershell
python -m pytest tests/test_batchnorm.py -q
python -m pytest tests/test_dropout.py -q
```

### 5단계: Network / Training / Evaluate

```powershell
python -m pytest tests/test_neural_network.py -q
python -m pytest tests/test_training.py -q
python -m pytest tests/test_evaluate.py -q
```

### 전체 테스트

```powershell
python -m pytest tests/ -v
```

---

## 10. MNIST 데이터 다운로드

데이터가 필요하면 프로젝트 루트에서 아래 명령을 실행한다.

```powershell
python download_mnist.py
```

생성되는 `data/mnist.npz`는 Git에 올리지 않는다.

---

## 11. 환경 비활성화와 삭제

작업을 마치고 환경을 비활성화하려면 아래 명령을 사용한다.

```powershell
conda deactivate
```

환경을 완전히 삭제하고 다시 만들고 싶을 때만 아래 명령을 사용한다.

```powershell
conda env remove -n mnist-nn
```

삭제 후 다시 만들기:

```powershell
conda create -n mnist-nn python=3.11 -y
conda activate mnist-nn
python -m pip install -r requirements.txt
```

---

## 12. 자주 확인할 명령

현재 Python:

```powershell
python --version
```

현재 Python 위치:

```powershell
python -c "import sys; print(sys.executable)"
```

Conda 환경 목록:

```powershell
conda env list
```

현재 환경의 패키지:

```powershell
python -m pip list
```

첫 테스트:

```powershell
python -m pytest tests/test_relu.py -q
```
