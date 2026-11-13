# 실행 환경과 검증 범위

2026-11-13 정리 환경: Windows, CPython 3.14.2. 흡연/난임 requirements는 현재 설치된 배포판 메타데이터의 버전을 고정했습니다. SmartDUR requirements는 원본 requirements-lock을 보존했습니다. 이는 과거 전체 실험의 완전한 환경 lock을 새로 검증한 것이 아닙니다.

Python 3.14 x64를 우선 권장합니다. requirements-dev는 정적 검사 도구이며 별도 설치합니다. 패키지 별 Python/Windows 휠 확인 결과는 dependency-check.json에 기록합니다. 메타데이터 확인은 전체 전이 의존성 해결이나 ABI/모델 재현의 보장이 아니며, 현재 PC에 없는 Keras·LightGBM·SHAP 등은 전체 import·학습 검증을 하지 않았습니다.

SmartDUR는 CPU로 고정하며 TensorFlow GPU 환경을 요구하지 않습니다. 원본에 기록된 CUDA 빌드와 동일한 GPU 환경을 설치할 필요는 없습니다. PyTorch CPU 배포를 쓰면 라이브러리 버전/결과 차이를 별도로 기록하세요.

공식 참고: [scikit-learn 설치](https://scikit-learn.org/stable/install), [CatBoost 설치](https://catboost.ai/docs/en/installation/python-installation-method-pip-install), [PyTorch](https://pypi.org/project/torch/2.9.1/).
