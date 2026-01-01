# verl 코드베이스 학습 계획

## 프로젝트 개요

**verl** (Volcano Engine Reinforcement Learning)은 ByteDance Seed 팀이 개발한 LLM을 위한 강화학습 훈련 라이브러리입니다.
- HybridFlow 논문의 오픈소스 구현체
- PPO, GRPO, RLOO, DAPO 등 다양한 RL 알고리즘 지원
- 단일 GPU부터 671B+ 모델까지 확장 가능
- vLLM, SGLang, Megatron-LM 등과 통합

---

## 학습 로드맵

### Phase 1: 기초 이해 (1-2일)

#### 1.1 핵심 개념 파악
- [ ] README.md 정독
- [ ] docs/ 디렉토리의 문서 읽기
- [ ] RLHF(Reinforcement Learning from Human Feedback) 개념 이해
- [ ] PPO(Proximal Policy Optimization) 알고리즘 이해

#### 1.2 프로젝트 구조 파악
```
verl/
├── trainer/          # 훈련 루프 구현 (핵심)
├── workers/          # 분산 훈련 워커
├── models/           # 모델 통합
├── single_controller/ # Ray 기반 오케스트레이션
├── protocol.py       # 데이터 전송 프로토콜
└── utils/            # 유틸리티
```

#### 1.3 의존성 기술 스택
- **Ray**: 분산 작업 실행
- **PyTorch FSDP**: 분산 훈련
- **Hydra**: 설정 관리
- **vLLM/SGLang**: 추론 백엔드
- **TensorDict**: 효율적 텐서 배칭

---

### Phase 2: 핵심 모듈 학습 (3-5일)

#### 2.1 데이터 프로토콜 (`verl/protocol.py`)
**우선순위: ★★★★★**

학습 포인트:
- [ ] `DataProto` 클래스 구조 이해
- [ ] 텐서 딕셔너리 기반 데이터 전송 방식
- [ ] 패딩 및 동기화 메커니즘

```python
# 핵심 파일
verl/protocol.py
```

#### 2.2 트레이너 모듈 (`verl/trainer/`)
**우선순위: ★★★★★**

학습 포인트:
- [ ] `main_ppo.py` - 메인 진입점
- [ ] `ppo/ray_trainer.py` - Ray 기반 분산 PPO 트레이너
- [ ] `ppo/core_algos.py` - GAE, advantage estimation 등 핵심 알고리즘
- [ ] `ppo/reward.py` - 보상 계산

```python
# 학습 순서
1. verl/trainer/main_ppo.py          # 진입점
2. verl/trainer/ppo/ray_trainer.py   # 오케스트레이션
3. verl/trainer/ppo/core_algos.py    # 알고리즘 구현
4. verl/trainer/config/              # 설정 구조
```

#### 2.3 워커 모듈 (`verl/workers/`)
**우선순위: ★★★★☆**

학습 포인트:
- [ ] `fsdp_workers.py` - FSDP 백엔드
- [ ] `megatron_workers.py` - Megatron 백엔드
- [ ] `rollout/vllm_rollout/` - vLLM 추론
- [ ] `rollout/sglang_rollout/` - SGLang 추론

```python
# 핵심 파일
verl/workers/fsdp_workers.py
verl/workers/rollout/vllm_rollout/
```

#### 2.4 분산 컨트롤러 (`verl/single_controller/`)
**우선순위: ★★★★☆**

학습 포인트:
- [ ] `RayWorkerGroup` - 워커 그룹 관리
- [ ] `RayResourcePool` - GPU 리소스 할당
- [ ] 워커 공존(colocate) 메커니즘

---

### Phase 3: 알고리즘 심화 (3-4일)

#### 3.1 PPO 구현 이해
- [ ] `verl/trainer/ppo/core_algos.py` 상세 분석
- [ ] GAE(Generalized Advantage Estimation) 구현
- [ ] KL 페널티 및 클리핑 메커니즘
- [ ] 가치 함수 손실 계산

#### 3.2 다른 알고리즘 비교
- [ ] GRPO (Group Relative Policy Optimization)
- [ ] RLOO (Reinforcement Learning with Likelihood-based Objective Optimization)
- [ ] DAPO (Diffusion-based Augmentation PPO)
- [ ] ReMax

```python
# 알고리즘별 예제
examples/ppo_trainer/
examples/grpo_trainer/
examples/rloo_trainer/
recipe/dapo/
```

---

### Phase 4: 실습 (2-3일)

#### 4.1 예제 실행
- [ ] SFT(Supervised Fine-Tuning) 예제 실행
- [ ] PPO 훈련 예제 실행
- [ ] GRPO 훈련 예제 실행

```bash
# 예제 스크립트 위치
examples/sft/gsm8k/
examples/ppo_trainer/run_qwen2-7b.sh
examples/grpo_trainer/run_qwen3-8b.sh
```

#### 4.2 설정 시스템 이해
- [ ] Hydra 설정 구조 분석
- [ ] YAML 설정 파일 수정해보기
- [ ] 커맨드 라인 오버라이드 테스트

---

### Phase 5: 고급 주제 (선택적)

#### 5.1 멀티모달 지원
- [ ] VLM(Vision-Language Model) 통합
- [ ] 이미지 기반 RL 훈련

#### 5.2 커스텀 확장
- [ ] 새로운 알고리즘 추가 방법
- [ ] 커스텀 리워드 모델 통합
- [ ] 새로운 백엔드 추가

#### 5.3 성능 최적화
- [ ] 시퀀스 길이 밸런싱
- [ ] 메모리 최적화 기법
- [ ] 분산 훈련 스케일링

---

## 핵심 파일 학습 순서

| 순서 | 파일 | 설명 |
|------|------|------|
| 1 | `verl/protocol.py` | 데이터 전송 프로토콜 |
| 2 | `verl/trainer/main_ppo.py` | 메인 진입점 |
| 3 | `verl/trainer/ppo/ray_trainer.py` | 분산 트레이너 |
| 4 | `verl/trainer/ppo/core_algos.py` | 핵심 알고리즘 |
| 5 | `verl/workers/fsdp_workers.py` | FSDP 워커 |
| 6 | `verl/workers/rollout/vllm_rollout/` | 추론 워커 |
| 7 | `verl/single_controller/ray/` | Ray 오케스트레이션 |
| 8 | `verl/trainer/config/` | 설정 시스템 |

---

## 추천 학습 자료

### 필수 개념
1. **RLHF**: InstructGPT 논문, Constitutional AI
2. **PPO**: Schulman et al., 2017 "Proximal Policy Optimization Algorithms"
3. **FSDP**: PyTorch Fully Sharded Data Parallel 문서
4. **Ray**: Ray Core 및 Ray Train 문서

### 프로젝트 문서
- `docs/` 디렉토리
- GitHub README
- HybridFlow 논문

---

## 학습 체크리스트

### Week 1
- [ ] Phase 1 완료 (기초 이해)
- [ ] Phase 2의 2.1, 2.2 완료 (protocol, trainer)

### Week 2
- [ ] Phase 2의 2.3, 2.4 완료 (workers, controller)
- [ ] Phase 3 시작 (알고리즘 심화)

### Week 3
- [ ] Phase 3 완료
- [ ] Phase 4 실습 진행

### Week 4 (선택적)
- [ ] Phase 5 고급 주제 탐구

---

## 질문 및 노트

학습 중 발생하는 질문과 메모를 여기에 기록하세요:

### 질문
1.

### 메모
1.

---

*마지막 업데이트: 2026-01-01*
