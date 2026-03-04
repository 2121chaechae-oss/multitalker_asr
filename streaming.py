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
from nemo.collections.asr.models import SortformerEncLabelModel

#스트리밍 추가
import pyaudio
import numpy as np
import sys
import torchaudio

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
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    cuda: Optional[int] = None
    allow_mps: bool = False
    matmul_precision: str = "highest"

    asr_model: Optional[str] = None
    device: str = "cuda"
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    batch_size: int = 32
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
# =============================================================================
# 설정값 
# =============================================================================
ASR_MODEL_PATH = os.environ.get("MULTITALKER_ASR_MODEL")
DIAR_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"

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
    
    if not ASR_MODEL_PATH:
        raise ValueError(
            "ASR model path is not set. Use --asr-model-path or set MULTITALKER_ASR_MODEL."
        )
    if not os.path.exists(ASR_MODEL_PATH):
        raise FileNotFoundError(f"ASR model file not found: {ASR_MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[INFO] Using device: {device}")

    # Diarization 모델 로드
    print("[INFO] Loading Diarization model (Sortformer)...")
    diar_model = SortformerEncLabelModel.from_pretrained(DIAR_MODEL_NAME)
    diar_model = diar_model.eval().to(device)
    # ASR 모델 로드
    print("[INFO] Loading ASR model (Multitalker Parakeet)...")
    asr_model = nemo_asr.models.ASRModel.restore_from(ASR_MODEL_PATH)  # <- 여기를 구현하세요
    asr_model = asr_model.eval().to(device)
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


# =============================================================================
# 설정 초기화 함수 구현
# =============================================================================
def setup_config(output_path: str = "streaming_results.json"): 
    # <- NeMo가 에러를 뱉지 않도록 가짜 파일명을 넣어준다.
    """
    멀티토커 ASR 설정을 초기화합니다.
    
    📌 구현해야 할 내용:
        1. MultitalkerTranscriptionConfig 객체 생성
        2. 필요한 설정값들 지정
    
    Args:
        audio_file (str): 입력 오디오 파일 경로
        output_path (str): 출력 JSON 파일 경로 (선택)
    
    Returns:
        OmegaConf: 설정 객체
    
    📌 주요 설정 항목:
        - audio_file: 처리할 오디오 파일 경로
        - output_path: 결과 저장 경로
        - max_num_of_spks: 최대 화자 수 (4)
        - streaming_mode: 스트리밍 모드 사용 여부 (True)
        - use_amp: Automatic Mixed Precision 사용 (True, GPU 메모리 절약)
        - masked_asr: False (Multitalker speaker-target 경로 사용)
        - binary_diar_preds: True (diarization mask 이진화)
    """

    # 설정 객체 생성 및 값 설정
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.output_path = output_path
    cfg.max_num_of_spks = 4  
    cfg.streaming_mode = True  # <- 스트리밍 모드
    cfg.online_normalization = False  # <- 온라인 정규화
    cfg.pad_and_drop_preencoded = False  # <- 패딩 설정
    cfg.use_amp = True  # <- Mixed Precision 사용
    cfg.colored_text = True  # <- 색상 출력
    cfg.masked_asr = False 
    cfg.binary_diar_preds = True
    return cfg
# =============================================================================
# 메인 추론 함수 구현
# =============================================================================
def run_multitalker_inference(output_path: str = None):
    """
    멀티토커 ASR 추론을 실행합니다.
    
    📌 전체 파이프라인:
        1. 모델 로드 (Diarization + ASR)
        2. 설정 초기화
        3. Diarization 모델 스트리밍 설정
        5. SpeakerTaggedASR 초기화
        6. 청크 단위 스트리밍 처리
        7. 최종 결과 생성 및 출력
    
    Args:
        audio_file (str): 입력 오디오 파일 경로 (16kHz WAV 권장)
        output_path (str): 출력 JSON 파일 경로 (선택)
    
    Returns:
        list: 화자별 인식 결과
    """
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
    # 모델 로드
    asr_model, diar_model, device = load_models()
    # 설정 초기화
    cfg = setup_config(output_path = "streaming_results.json")
    
    # Diarization 모델 스트리밍 설정
    # 스트리밍 처리를 위한 내부 버퍼 및 상태 초기화
    print("[INFO] Initializing diarization model for streaming...")
    diar_model = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar_model)


    # SpeakerTaggedASR 초기화
    # Diarization 결과를 기반으로 화자별 ASR을 수행하는 헬퍼 클래스
    # 내부적으로 화자 수만큼 ASR 인스턴스 관리
    print("[INFO] Setting up SpeakerTaggedASR...")
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)  # <- 여기를 구현하세요
    
    # =========================================================================
    # Step 6: 스트리밍 처리 루프
    # 
    # 📌 처리 흐름:
    #     for each chunk in streaming_buffer:
    #         1. Diarization: 이 청크에서 누가 말하는지 감지
    #         2. ASR: 각 화자별로 음성 인식 수행
    #         3. 중간 결과 업데이트
    # =========================================================================
    
    # 스트리밍 추가
    # 마이크 장치 설정
    MIC_RATE = 48000
    MODEL_RATE = 16000       # 모델이 인식하는 표준 주파수 (16kHz)
    # chunk_size는 모델의 설정(cfg)에 따라 달라질 수 있습니다. 
    # 보통 NeMo 스트리밍은 0.4초~0.8초 단위로 처리합니다. (16000 * 0.4 = 6400)
    # 만약 에러가 난다면 스켈레톤 코드의 chunk_size 변수 값을 여기에 넣어주세요.
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 19200
    
    # 마이크/스피커 연결
    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = MIC_RATE,
                    input = True,
                    input_device_index=15, # DJI 마이크 번호
                    frames_per_buffer=CHUNK) 
    resampler = torchaudio.transforms.Resample(orig_freq=MIC_RATE, new_freq=MODEL_RATE).to(device)
    print("\n" + "="*60)
    print(f"[INFO] 마이크가 켜졌습니다. 종료하려면 Ctrl+C를 누르세요.")
    print("="*60 + "\n")    
    

    
    step_num = 0
    try :
        while True:
            # 마이크에서 소리 조각 가져오기
            data = stream.read(CHUNK, exception_on_overflow = False)
            # 파이토치 텐서 ([1, 길이] 형태의 float32)
            audio_signal = np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0
            
            # 마이크에 실제로 소리가 들어가고 있는지 확인
            audio_signal = audio_signal * 15.0
            if step_num % 10 == 0:
                max_vol = np.max(np.abs(audio_signal))
                # 잔상 제거 코드 위쪽으로 예쁘게 출력되도록 \n 추가
                sys.stdout.write(f"\n🎤 [마이크 확인] 현재 소리 크기: {max_vol:.4f}\n")
            
            
            # 텐서 변환
            audio_tensor = torch.tensor(audio_signal).unsqueeze(0).to(device)
            resampled_tensor = resampler(audio_tensor)
            audio_len = torch.tensor([resampled_tensor.shape[1]]).to(device)
            
            #모델의 전처리기를 직접 호출해서 데이터를 가공한다. raw audio -> mel spectrogram
            processed_signal, processed_len = asr_model.preprocessor(input_signal=resampled_tensor, length=audio_len)
            
            # 드롭 설정 (스켈레톤 코드 그대로)
            drop_extra_pre_encoded = (
                0
                if step_num ==0 and not cfg.pad_and_drop_preencoded
                else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
            )
                    
        # 스트리밍 ASR 처리
            with torch.inference_mode():
                with torch.amp.autocast(device.type, enabled=cfg.use_amp):
                    with torch.no_grad():
                        multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                            step_num=step_num,
                            chunk_audio=processed_signal,
                            chunk_lengths=processed_len,
                            is_buffer_empty=False, # 마이크가 계속 켜져 있으니 버퍼에도 뭔가가 계속 담겨 있다.
                            drop_extra_pre_encoded=drop_extra_pre_encoded
                        )
            live_display=[]
            if hasattr(multispk_asr_streamer, 'active_previous_hypotheses'):
                active_hyps = multispk_asr_streamer.active_previous_hypotheses
                
                if isinstance(active_hyps, list):
                    for i, hyp in enumerate(active_hyps):
                        if hyp is not None:
                            # 텍스트 추출 (객체일 수도 있고 문자열일 수도 있음.)
                            text = hyp.y_sequence if hasattr(hyp, 'y_sequence') else str(hyp)
                            if text.strip() and text != "None":
                                live_display.append(f"[화자{i}]: {text}")
                
                elif isinstance(active_hyps, dict):
                    for spk_id, hyp in active_hyps.items():
                        if hyp is not None:
                            text = hyp.y_sequence if hasattr(hyp, 'y_sequence') else str(hyp)
                            if text.strip() and text !="None":
                                live_display.append(f"[{spk_id}]: {text}")
                    
                            
                                        
            #터미널 출력 (잔상 제거 포함)
            if live_display:
                sys.stdout.write("f\r[실시간] {'|'.join(live_display)}" + " "*20)
            else:
                sys.stdout.write(f"\r[대기중] 소리 분석 중..." + " "*20)
            sys.stdout.flush()    
            step_num +=1
        
    except KeyboardInterrupt:
        print("\n 마이크 스트리밍을 종료합니다.")
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        

    print(f"\n[INFO] Processing completed! Total steps: {step_num}")
    print("\n[INFO] Generating final transcripts...")

    # 결과 가져오기
    results = multispk_asr_streamer.instance_manager.seglst_dict_list
    # 결과 출력 (수정하지 마세요)
    print_results(results, multispk_asr_streamer, cfg.output_path)
    
    return results

# JSON 저장
def print_results(results, multispk_asr_streamer, output_path):
        
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[INFO] Results saved to: {output_path}")



# =============================================================================
# 메인 실행부 (수정하지 마세요)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multitalker ASR Inference")
    parser.add_argument(
        "--asr-model-path",
        type=str,
        default=os.environ.get("MULTITALKER_ASR_MODEL"),
        help="Path to multitalker-parakeet-streaming-0.6b-v1.nemo",
    )
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.audio_file):
        args.audio_file = os.path.join(os.getcwd(), args.audio_file)

    if not args.asr_model_path:
        parser.error(
            "ASR model path is required. Pass --asr-model-path or set MULTITALKER_ASR_MODEL."
        )

    ASR_MODEL_PATH = args.asr_model_path
    run_multitalker_inference(args.audio_file, args.output)
