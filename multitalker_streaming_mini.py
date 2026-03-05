#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
[과제] 멀티토커 ASR 시스템 구현
================================================================================

📌 과제 목표:
    여러 화자가 동시에 말하는 (오버랩) 오디오에서 각 화자별로 음성을 인식하는
    시스템을 구현합니다.

📌 시스템 구조:
    ┌─────────────────────────────────────────────────────────────┐
    │                    입력 오디오 (오버랩 포함)                   │
    └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │         [Step 1] Speaker Diarization (화자 분리)             │
    │         "누가 언제 말하는지" 감지 → 화자 타깃 정보 생성        │
    └─────────────────────────────────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
            ┌──────────┐    ┌──────────┐    ┌──────────┐
            │ [Step 2] │    │ [Step 2] │    │ [Step 2] │
            │ ASR #1   │    │ ASR #2   │    │ ASR #3   │
            │ Speaker0 │    │ Speaker1 │    │ Speaker2 │
            └──────────┘    └──────────┘    └──────────┘
                  │               │               │
                  ▼               ▼               ▼
            "안녕하세요"    "네 반갑습니다"    "저도요"

📌 사용 모델:
    1. Diarization: nvidia/diar_streaming_sortformer_4spk-v2.1
       - Sortformer 기반 스트리밍 화자 분리 모델
       - 최대 4명의 화자 지원
    
    2. ASR: multitalker-parakeet-streaming-0.6b-v1
       - Fast-Conformer 인코더 + RNN-T 디코더
       - Speaker Kernel Injection으로 특정 화자에 집중

📌 핵심 개념:
    - Speaker Diarization: 화자 분리 (누가 언제 말하는지)
    - Speaker Target Conditioning: diarization 출력을 이용해 화자별 인식 경로 제어
    - Multi-Instance ASR: 화자 수만큼 ASR 모델 인스턴스 생성
    - Streaming: 청크 단위로 실시간 처리

📌 참고 자료:
    - HuggingFace: https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1
    - NeMo Framework: https://github.com/NVIDIA/NeMo

================================================================================
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from omegaconf import OmegaConf, open_dict

# NeMo imports
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR

#스트리밍 추가
import pyaudio
import numpy as np
import sys
import torchaudio
import json

# 정보성 로그 무시
import logging
from nemo.utils import logging as nemo_logging
nemo_logging.set_verbosity(nemo_logging.ERROR)
# =============================================================================
# 기본 제공 Config 클래스
# =============================================================================
@dataclass
class MultitalkerTranscriptionConfig:
    diar_model: Optional[str] = None
    diar_pretrained_name: Optional[str] = None
    max_num_of_spks: Optional[int] = 4
    parallel_speaker_strategy: bool = True
    masked_asr: bool = True
    mask_preencode: bool = False
    cache_gating: bool = True
    cache_gating_buffer_size: int = 2
    single_speaker_mode: bool = False

    session_len_sec: float = -1
    num_workers: int = 8
    random_seed: Optional[int] = None
    log: bool = True

    streaming_mode: bool = True
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 14
    chunk_left_context: int = 0
    chunk_right_context: int = 13

    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"

    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    batch_size: int = 1
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    generate_realtime_scripts: bool = False

    word_window: int = 50
    sent_break_sec: float = 30.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None

    ignored_initial_frame_steps: int = 5
    verbose: bool = False

    feat_len_sec: float = 0.01
    finetune_realtime_ratio: float = 0.01

    spk_supervision: str = "diar"
    binary_diar_preds: bool = False

    @staticmethod
    def init_diar_model(cfg, diar_model):
        diar_model.streaming_mode = cfg.streaming_mode
        diar_model.sortformer_modules.chunk_len = cfg.chunk_len if cfg.chunk_len > 0 else 6
        diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
        diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        diar_model.sortformer_modules.log = cfg.log
        diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
        return diar_model
# 화자별 출력 색상
SPEAKER_COLORS = {
    0: "\033[94m",   # Blue
    1: "\033[92m",   # Green  
    2: "\033[93m",   # Yellow
    3: "\033[95m",   # Magenta
}
RESET_COLOR = "\033[0m"
# =============================================================================
# 모델 로드 함수 구현
# =============================================================================
def load_models(device: str = "cuda"):
    """
    Diarization 모델과 ASR 모델을 로드합니다.
    
    📌 구현해야 할 내용:
        1. GPU/CPU 디바이스 설정
        2. Diarization 모델 로드 (SortformerEncLabelModel.from_pretrained)
        3. ASR 모델 로드 (nemo_asr.models.ASRModel.restore_from)
        4. CUDA Graph 비활성화 (스트리밍 호환성을 위해)
    
    Args:
        device (str): 사용할 디바이스 ("cuda" 또는 "cpu")
    
    Returns:
        tuple: (asr_model, diar_model, device)
    
    Hints:
        - torch.cuda.is_available()로 GPU 사용 가능 여부 확인
        - device는 torch.device(...) 형태로 만들어야 함 (아래 autocast에서 device.type 사용)
        - 모델 로드 후 .eval()과 .to(device) 호출 필요
        - CUDA Graph 비활성화는 _disable_cuda_graph() 함수 참고
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[INFO] Using device: {device}")

    # Diarization 모델 로드
    print("[INFO] Loading Diarization model (Sortformer)...")
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1").eval().to(device)
    # ASR 모델 로드
    print("[INFO] Loading ASR model (Multitalker Parakeet)...")
    asr_model = ASRModel.restore_from("./model/multitalker-parakeet-streaming-0.6b-v1.nemo").eval().to(device)
    # CUDA Graph 비활성화
    asr_model = _disable_cuda_graph(asr_model)
    print("[INFO] Models loaded successfully!")
    return asr_model, diar_model, device


def _disable_cuda_graph(asr_model):
    """
    CUDA Graph 디코더 비활성화 (수정하지 마세요)
    
    📌 왜 필요한가?
        - CUDA Graph는 연산 그래프를 미리 컴파일하여 오버헤드를 줄임
        - 하지만 입력 크기가 고정되어야 함
        - 스트리밍에서는 청크 크기가 가변적일 수 있어 비활성화 필요
    """
    with open_dict(asr_model.cfg):
        if hasattr(asr_model.cfg, 'decoding'):
            asr_model.cfg.decoding.greedy.loop_labels = False
            asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    
    decoding_cfg = OmegaConf.create({
        'strategy': 'greedy',
        'greedy': {
            'max_symbols': 10,
            'loop_labels': False,
            'use_cuda_graph_decoder': False,
        }
    })
    asr_model.change_decoding_strategy(decoding_cfg)
    
    return asr_model



# 마이크 입력을 diar_model과 asr_model과 연결


if __name__ =="__main__":
    asr_model, diar_model, device = load_models()
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.audio_file = "dummy.wav"
    diar_model = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar_model)
    print("[INFO] Initializing SpeakerTaggedASR Streamer.")
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 17920 # 16000*1.12 = 17920 samples
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=48000,
                    input=True,
                    input_device_index=11,  # 마이크 장치 인덱스 (환경에 따라 다를 수 있음)
                    frames_per_buffer = CHUNK_SIZE*3)
    print("[INFO] 마이크 입력을 시작합니다. Ctrl+C 로 종료할 수 있습니다.")
    
    step_num = 0
    final_result=[]
    try:
        while True:
            data = stream.read(CHUNK_SIZE*3, exception_on_overflow=False)
            audio_np = np.frombuffer(data,dtype=np.int16).astype(np.float32)/32768.0 
            # 데이터 변환(Bytes-> numpy-> float32-> tensor)
            audio_np = audio_np[0::2]  # 스테레오에서 왼쪽 채널만 사용 (환경에 따라 조정 필요)
            audio_np = audio_np[0::3]
            raw_audio = torch.from_numpy(audio_np).unsqueeze(0).to(device)
            raw_lengths = torch.tensor([raw_audio.shape[1]]).to(device)
            
            with torch.inference_mode():
                with torch.amp.autocast(diar_model.device.type, enabled= True):
                    
                    chunk_features, chunk_feature_lengths = asr_model.preprocessor(
                        input_signal=raw_audio,
                        length = raw_lengths,
                    )
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio = chunk_features,
                        chunk_lengths = chunk_feature_lengths,
                        is_buffer_empty = False,
                        drop_extra_pre_encoded = 0 if step_num==0 else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded,               
                    )
            current_result = multispk_asr_streamer.instance_manager.batch_asr_states[0].seglsts
            if current_result:
                final_result = current_result
                print(f"[{step_num * 1.12: 2f}s] {current_result}")
            step_num += 1
        
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 스트리밍이 종료됩니다.")
        
        if final_result:
            print(f"[INFO] 인식 결과를 streaming_results.json 파일에 저장합니다.")
            with open("streaming_results.json", "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=4)
        else :
            print("[INFO] 저장할 음성 인식 결과가 없습니다.")
            
    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류가 발생했습니다. : {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("[INFO] 오디오 장치를 안전하게 닫았습니다.")
        