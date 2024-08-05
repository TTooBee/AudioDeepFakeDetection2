import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt

def poly2lsf(a):
    """
    Convert prediction polynomial to line spectral frequencies (LSF).
    
    Parameters:
    a (array-like): Prediction polynomial coefficients
    
    Returns:
    lsf (numpy.ndarray): Line spectral frequencies
    """
    # Ensure the input is a numpy array
    a = np.asarray(a)
    
    if a.ndim != 1:
        raise ValueError("Input polynomial must be a 1-D array.")
    
    if not np.isrealobj(a):
        raise ValueError("Input polynomial must be real.")
    
    # Normalize the polynomial if the first coefficient is not unity
    if a[0] != 1.0:
        a = a / a[0]
    
    # Check if the roots are within the unit circle
    if np.max(np.abs(np.roots(a))) >= 1.0:
        raise ValueError("Polynomial must have all roots inside the unit circle.")
    
    # Form the sum and difference filters
    p = len(a) - 1  # The leading one in the polynomial is not used
    a1 = np.append(a, 0)
    a2 = a1[::-1]
    P1 = a1 - a2  # Difference filter
    Q1 = a1 + a2  # Sum filter
    
    # If order is even, remove the known root at z = 1 for P1 and z = -1 for Q1
    if p % 2 != 0:  # Odd order
        P = np.polynomial.polynomial.polydiv(P1, [1, 0, -1])[0]
        Q = Q1
    else:  # Even order
        P = np.polynomial.polynomial.polydiv(P1, [1, -1])[0]
        Q = np.polynomial.polynomial.polydiv(Q1, [1, 1])[0]
    
    rP = np.roots(P)
    rQ = np.roots(Q)
    
    # Considering complex conjugate roots along with zeros for finding angles
    aP = np.angle(rP)
    aQ = np.angle(rQ)
    
    # Combine and sort the angles
    lsf_temp = np.sort(np.concatenate((aP, aQ)))
    
    # Remove negative angles and sort again
    lsf_temp = np.sort(lsf_temp[lsf_temp >= 0])
    
    # Ensure we return the correct number of LSFs
    lsf = lsf_temp[:p]
    
    return lsf

# Example usage
a = [1.0000, -1.6660, 0.9526, -0.2544, -0.7324, 1.3015, -0.6141, -0.2870, 0.6757, -0.5656, 0.2803, 0.0373, 0.0713, 0.0208, -0.2092, 0.0253, -0.0807, -0.0200, 0.2397, -0.1288, 0.0230]
lsf = poly2lsf(a)
print("Line spectral frequencies:", lsf)

# 오디오 파일 읽기
x, Fs = librosa.load("real_temp/wav/LJ001-0001_16k.wav", sr=None)

# 프레임 크기 및 시작 지점 설정
frame_size = 320
bp = 22000
M = 20

# 특정 구간 추출
segment = x[bp:bp + frame_size]

# segment의 시간 축 생성
time = np.arange(len(segment)) / Fs

# segment 플롯
plt.figure(figsize=(14, 5))
plt.plot(time, segment)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Waveform Segment of LJ001-0001_16k.wav')
plt.grid()

# 플롯을 PNG 파일로 저장
plt.savefig("segment_waveform.png")
plt.close()

# LPC 계수 계산
a = librosa.lpc(segment, order=M)

print("LPC coefficients:", a)
