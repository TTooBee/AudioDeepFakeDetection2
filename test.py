import os
import numpy as np
import librosa

# LPC를 LSF로 변환하는 함수
def lpc_to_lsf(lpc_coeffs):
    a = np.append([1], -lpc_coeffs)
    p = len(lpc_coeffs)
    b = np.zeros(p + 1)
    b[::2] = a[::2]
    b[1::2] = -a[1::2]
    
    roots_a = np.roots(a)
    roots_b = np.roots(b)
    
    angles_a = np.angle(roots_a)
    angles_b = np.angle(roots_b)
    
    angles_a = angles_a[np.isreal(roots_a)]
    angles_b = angles_b[np.isreal(roots_b)]
    
    lsf = np.sort(np.concatenate((angles_a, angles_b)))
    return lsf[:p]

# 오디오 파일 로드 및 프레임 나누기
def process_audio(file_path, frame_length=0.02, hop_length=0.01, order=16):
    # 절대 경로로 변환
    abs_file_path = os.path.abspath(file_path)
    
    # 파일 존재 여부 확인
    if not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"File not found: {abs_file_path}")
    
    y, sr = librosa.load(abs_file_path, sr=None)
    frame_size = int(frame_length * sr)
    hop_size = int(hop_length * sr)
    
    frames = librosa.util.frame(y, frame_length=frame_size, hop_length=hop_size).T
    
    lpc_coefficients = []
    lsf_frequencies = []
    
    for frame in frames:
        lpc_coeffs = librosa.lpc(frame, order=order)
        lpc_coefficients.append(lpc_coeffs)
        lsf_freqs = lpc_to_lsf(lpc_coeffs[1:])  # LPC 함수에서 첫 번째 계수는 제외
        lsf_frequencies.append(lsf_freqs)
    
    return np.array(lpc_coefficients), np.array(lsf_frequencies)

# 메인 함수
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <audio_file_path>")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]
    
    try:
        lpc_coeffs, lsf_freqs = process_audio(audio_file_path)
        
        # 결과 출력
        print("LPC Coefficients:")
        print(lpc_coeffs)
        print("\nLSF Frequencies:")
        print(lsf_freqs)
    except FileNotFoundError as e:
        print(e)
