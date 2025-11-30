# 🧠 Python Coding Test Answer Generator

> LeetCode 문제를 입력하면, 가독성과 효율성이 높은 파이썬 모범답안을 생성하는 sLLM 파인튜닝 프로젝트

---

## 📌 프로젝트 개요

### 배경
  코딩테스트는 소프트웨어 엔지니어 채용에서 핵심적인 평가 요소이며, LeetCode와 같은 온라인 저지의 문제들이 사실상 표준처럼 사용되고 있습니다.
### 목표
  LeetCode 문제 및 풀이 데이터를 활용해 “파이썬 기반 코딩테스트 모범답안 생성모델" 개발
  단순히 정답만 맞히는 것이 아니라, 가독성(PEP8)과 효율성(time/memory) 을 함께 고려한 코드 생성이 목표
### 핵심 기술 스택

  * sLLM 파인튜닝
  * SFT (Supervised Fine-Tuning)
  * DPO (Direct Preference Optimization)
  * (선택) ORPO, KTO, GRPO 등 최신 preference alignment 기법

---

## 📂 데이터셋

| 데이터셋                 | 설명                           | 출처                                                                                     |
| -------------------- | ---------------------------- | -------------------------------------------------------------------------------------- |
| **LeetCode Dataset** | 문제 설명 + 제약 조건 → Python 정답 코드 | [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset) |
| **APPS (보조 데이터)**    | 프로그래밍 문제와 정답 코드 쌍            | [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps)                     |

---

## 🛠️ 기술 스택

* **Framework**: PyTorch, Hugging Face Transformers
* **Training**: PEFT, TRL
* **Optimization**: LoRA / QLoRA, Flash-Attn, Gradient Checkpointing
* **Deployment**: vLLM (Page Attention), KV Cache
* **Evaluation**: Gemini / GPT 기반 코드 품질 평가

---

## 🧩 방법론

### 1. Supervised Fine-Tuning (SFT)

* 입력: LeetCode 문제 설명
* 출력: 정답 코드 + 코드 설명
* 모델이 출력 패턴과 함수 구조를 학습

### 2. Direct Preference Optimization (DPO)

* 동일 문제에 대한 두 개의 답안 코드 비교

  * 예: 정답이지만 비효율적인 코드 vs 더 최적화된 코드
* 모델이 완성도 높은 코드를 선택하도록 학습

### 3. ORPO (Optional)

* Odds Ratio 기반 preference 학습
* 가독성 및 코드 간결성 기준으로 선택 학습
* SFT & DPO를 하나의 training process에서 해결

---

## ⚙️ 실험 환경

| 항목          | 구성                                         |
| ----------- | ------------------------------------------ |
| **GPU**     | RunPod A100 (40GB)                         |
| **Python**  | 3.11.10                                    |
| **PyTorch** | 2.4                                        |
| **Model**   | `Qwen2.5-3B-Instruct` |
| **지원 기능**   | QLoRA, Flash-Attn, Gradient Checkpointing  |

---

## 🔍 참고 자료

* [TRL - Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)
* [LLM RLHF 기법 정리 (PPO, DPO, IPO, KTO, ORPO, GRPO)](https://davidlds.tistory.com/100)
* [A Systematic Survey of Prompt Engineering (2024)](https://arxiv.org/abs/2402.07927)
* [ORPO 논문 리뷰](https://meanwo0603.tistory.com/entry/ORPO-Monolithic-Preference-Optimization-without-Reference-Model-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)
* [RunPod 서버 대여 및 VS CODE 연동](https://velog.io/@lse0912/RunPod-%EC%84%9C%EB%B2%84-%EB%8C%80%EC%97%AC-%EB%B0%8F-VS-CODE-%EC%97%B0%EB%8F%99#runpod%EB%9E%80)

---

## 🧩 Repository Structure

```bash
.
├── checkpoints         # weights of trained model
├── config.json         
├── main.ipynb          # pipeline from data preparation to SFT/DPO
├── make_prompts.py     # prompts for training
├── qlora.py            # includes functions to load model and tokenizer for QLoRA manner
└── readme.md
```