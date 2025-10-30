import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pretty_midi
import io
from pathlib import Path
import tempfile

st.set_page_config(page_title='AI 쌤 가창 코치', page_icon='🎶')

st.title("AI 쌤과 함께하는 딩동댕 가창 코치")
st.markdown(
    """
    학생들이 업로드한 가창 녹음(MP3, WAV, M4A 등)을 정답 MIDI(선택)와 비교하여
    음정(pitch)·박자(timing) 정확도를 시각화하고 간단한 피드백을 제공합니다.

    사용법:
    1) 학생 녹음 파일 업로드
    2) (선택) 정답 MIDI 업로드 — 없으면 자동 추출된 멜로디를 정답으로 사용
    3) 분석 버튼을 눌러 결과 확인
    """
)

st.sidebar.header("기능 요약")
st.sidebar.markdown(
    "- 음정 추적: librosa.pyin (없으면 yin) 사용\n"
    "- 박자(온셋) 비교: librosa.onset & MIDI onset 비교\n"
    "- 간단한 정확도 점수(음정/박자) 및 피치 트랙 시각화\n\n"
    "필수 라이브러리: librosa, pretty_midi, matplotlib"
)

uploaded_audio = st.file_uploader("학생 녹음 파일 업로드 (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
uploaded_midi = st.file_uploader("정답 MIDI 파일 업로드 (선택)", type=["mid", "midi"])

# 재생 기능: 업로드 후 브라우저에서 바로 재생
if uploaded_audio:
    st.caption(f"업로드된 파일: {uploaded_audio.name}")
    st.audio(uploaded_audio)  # 업로드된 파일의 바이트를 전달하여 브라우저에서 재생

threshold_cents = st.slider("음정 정확도 허용 범위 (cent)", min_value=10, max_value=200, value=50, step=5)
onset_tolerance = st.slider("박자(온셋) 허용 오차 (초)", min_value=0.01, max_value=0.5, value=0.12, step=0.01)

@st.cache_data
def load_audio_file(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # librosa can usually read mp3/m4a via soundfile/audioread; ensure a stable path
        tmp.write(file_bytes.getvalue())
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    return y, sr

@st.cache_data
def extract_pitch(y, sr):
    # Try pYIN first (librosa.pyin), fallback to yin
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                    fmax=librosa.note_to_hz('C7'))
    except Exception:
        # fallback
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_flag = ~np.isnan(f0)
    times = librosa.times_like(f0, sr=sr)
    return times, f0, voiced_flag

@st.cache_data
def midi_to_reference(midi_bytes, sr=22050):
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes.getvalue()))
    # Create a time grid and synthesize reference pitch (monophonic melody: highest piano roll per frame)
    duration = pm.get_end_time()
    times = np.linspace(0, duration, int(duration * sr // 256) + 1)  # coarse grid
    ref_freq = np.full_like(times, np.nan, dtype=float)
    for inst in pm.instruments:
        for note in inst.notes:
            idx = np.where((times >= note.start) & (times <= note.end))
            ref_freq[idx] = pretty_midi.note_number_to_hz(note.pitch)
    # If no MIDI notes found, return empty
    return times, ref_freq

def freq_to_cents(f_a, f_b):
    return 1200 * np.log2(f_a / f_b)

def compute_pitch_accuracy(student_times, student_f0, ref_times, ref_f0, cents_thresh):
    # Interpolate reference to student times
    ref_interp = np.interp(student_times, ref_times, np.nan_to_num(ref_f0, nan=0.0))
    mask_voiced = ~np.isnan(student_f0) & (student_f0 > 0) & (ref_interp > 0)
    if mask_voiced.sum() == 0:
        return 0.0, []
    cents_diff = np.abs(1200 * np.log2(student_f0[mask_voiced] / ref_interp[mask_voiced]))
    accurate = cents_diff <= cents_thresh
    accuracy = accurate.sum() / len(accurate) * 100.0
    # return accuracy and list of (time, cents_diff) for voiced frames
    times = student_times[mask_voiced]
    return accuracy, list(zip(times.tolist(), cents_diff.tolist(), accurate.tolist()))

def compute_onset_accuracy(y, sr, ref_onsets, tol):
    # detect onsets in student's audio
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
    if len(ref_onsets) == 0:
        return 0.0, onset_frames, ref_onsets
    matched = 0
    used_ref = []
    for ro in ref_onsets:
        # if any detected onset within tol of ro
        if np.any(np.abs(onset_frames - ro) <= tol):
            matched += 1
            used_ref.append(ro)
    accuracy = matched / len(ref_onsets) * 100.0
    return accuracy, onset_frames, np.array(used_ref)

def midi_onsets_from_midi(pm):
    onsets = []
    for inst in pm.instruments:
        for note in inst.notes:
            onsets.append(note.start)
    return np.array(sorted(onsets))

# Main action
if st.button("분석 시작"):

    if not uploaded_audio:
        st.warning("먼저 학생 녹음 파일을 업로드해주세요.")
        st.stop()

    with st.spinner("오디오 로드 및 피치 추출 중..."):
        y, sr = load_audio_file(uploaded_audio)
        student_times, student_f0, voiced_flag = extract_pitch(y, sr)

    ref_times = None
    ref_f0 = None
    ref_onsets = np.array([])

    if uploaded_midi:
        try:
            pm = pretty_midi.PrettyMIDI(io.BytesIO(uploaded_midi.getvalue()))
            ref_onsets = midi_onsets_from_midi(pm)
            # create ref time grid matching student_times for easier plotting/comparison
            ref_times = student_times
            ref_f0 = np.full_like(ref_times, np.nan, dtype=float)
            for inst in pm.instruments:
                for note in inst.notes:
                    idx = (ref_times >= note.start) & (ref_times <= note.end)
                    ref_f0[idx] = pretty_midi.note_number_to_hz(note.pitch)
        except Exception as e:
            st.warning(f"MIDI 파싱 실패: {e}. 자동 추출 멜로디로 대체합니다.")
            uploaded_midi = None

    if not uploaded_midi:
        # Try simple harmonic/percussive separation + librosa.harmonic -> pitch as ref
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # use student's harmonic component as approximate reference (useful when no MIDI)
        y_harmonic = librosa.effects.harmonic(y)
        ref_times_alt, ref_f0_alt, _ = extract_pitch(y_harmonic, sr)
        ref_times = ref_times_alt
        ref_f0 = ref_f0_alt
        ref_onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    # Compute pitch accuracy
    pitch_acc, pitch_details = compute_pitch_accuracy(student_times, student_f0, ref_times, ref_f0, threshold_cents)

    # Compute onset/timing accuracy
    onset_acc, detected_onsets, matched_ref_onsets = compute_onset_accuracy(y, sr, ref_onsets, onset_tolerance)

    # Summary
    st.subheader("요약 결과")
    st.metric("음정 정확도", f"{pitch_acc:.1f} %")
    st.metric("박자(온셋) 정확도", f"{onset_acc:.1f} %")

    # Highlight worst segments (lowest accuracy frames)
    if pitch_details:
        # pitch_details: list of (time, cents_diff, accurate_bool)
        bad = [d for d in pitch_details if not d[2]]
        if bad:
            # show up to 3 worst by cents diff
            bad_sorted = sorted(bad, key=lambda x: -x[1])[:3]
            st.markdown("#### 취약 구간(음정)")
            for t, cents, _ in bad_sorted:
                st.write(f"- {t:.2f}s — 편차 {cents:.1f} cent (권장 연습: 해당 구역의 느린 템포 반복 연습)")
        else:
            st.write("음정이 전반적으로 안정적입니다.")

    # Plot pitch tracks
    st.subheader("피치 트래킹 (정답 vs 학생)")
    st.markdown(
        "정답: 주황색 실선 — MIDI(또는 자동 추출된 기준 멜로디)\n\n"
        "학생: 파란색 점 — 업로드한 녹음에서 추출한 피치\n\n"
        "빨간 × 표시는 설정한 허용 범위(cent)를 벗어난 취약 음정 구간입니다."
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))

    # 학생 f0 (파란 점)
    ax.plot(student_times, student_f0, '.', markersize=3, color='tab:blue',
            alpha=0.8)  # 레전드 제거: label 사용 안함

    # 정답 f0 (주황 실선)
    if ref_f0 is not None:
        ax.plot(ref_times, ref_f0, '-', linewidth=1.5, color='tab:orange',
                alpha=0.95)  # 레전드 제거

    # 취약 음정 표시 (빨간 ×)
    if pitch_details:
        bad = [d for d in pitch_details if not d[2]]
        if bad:
            times_bad = [d[0] for d in bad]
            y_bad = np.interp(times_bad, student_times, np.nan_to_num(student_f0, nan=0.0))
            ax.scatter(times_bad, y_bad, color='red', marker='x', s=40)

    ax.set_xlabel("시간 (s)")

    # Y축: Hz 대신 음이름(C4 등)으로 표시
    def midi_to_name(m):
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{names[int(m) % 12]}{int(m) // 12 - 1}"

    # 학생/정답 데이터에서 유효한 주파수 추출 (0 또는 NaN 제외)
    def valid_hz(arr):
        if arr is None:
            return np.array([])
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        a = a[a > 0]
        return a

    hz_student = valid_hz(student_f0)
    hz_ref = valid_hz(ref_f0) if 'ref_f0' in locals() and ref_f0 is not None else np.array([])
    all_hz = np.concatenate([hz_student, hz_ref]) if len(hz_student) + len(hz_ref) > 0 else np.array([440.0])

    min_hz = max(50.0, np.nanmin(all_hz) * 0.9)
    max_hz = min(20000.0, np.nanmax(all_hz) * 1.1)

    # MIDI 범위 (소수점 포함)으로 변환하고 적절한 C(옥타브) 위치를 선택
    midi_min = int(np.floor(librosa.hz_to_midi(min_hz)))
    midi_max = int(np.ceil(librosa.hz_to_midi(max_hz)))
    # C 음(페달 기준)만 표시: MIDI 번호 % 12 == 0 인 값들
    c_notes = [n for n in range(midi_min - 1, midi_max + 1) if n % 12 == 0]
    if not c_notes:
        c_notes = list(range(midi_min, midi_max + 1, 12))

    hz_ticks = librosa.midi_to_hz(np.array(c_notes))
    labels = [midi_to_name(n) for n in c_notes]

    # 안전 장치: tick 위치가 y 범위에 들어오도록 제한
    hz_ticks_in_range = [h for h in hz_ticks if (h >= min_hz and h <= max_hz)]
    labels_in_range = [labels[i] for i, h in enumerate(hz_ticks) if (h >= min_hz and h <= max_hz)]

    if hz_ticks_in_range:
        ax.set_yticks(hz_ticks_in_range)
        ax.set_yticklabels(labels_in_range)
    else:
        # fallback: 기본 Hz 눈금 유지하되 소수 자릿수 줄임
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    ax.set_ylim(min_hz, max_hz)
    ax.grid(axis='x', linestyle=':', alpha=0.3)

    # 레전드(오른쪽 상단 표시) 제거 — 사용자 요청에 따라 빈 공간으로 둠
    # ax.legend(...) 호출하지 않음

    st.pyplot(fig)

    # 설명 텍스트 (명확하게 어떤 선/점이 어떤 데이터인지)
    st.markdown(
        "- 파란 점: 학생 녹음에서 추출한 실제 발성 피치\n"
        "- 주황 실선: 교사가 제공한 MIDI(또는 자동 추출한 기준 멜로디)\n"
        "- 빨간 ×: 학생 피치가 설정한 cent 허용 범위를 벗어난 구간\n"
    )

    # Show onset comparison plot
    st.subheader("온셋(박자) 비교")
    st.markdown(
        "파형 위의 세로선으로 박자(온셋)를 비교합니다.\n\n"
        "- 파란 세로선: 학생의 녹음에서 감지된 온셋\n"
        "- 초록 점선: 정답 MIDI(또는 자동 추출) 온셋\n"
        "시간 오차 허용 범위 내에 있으면 박자 정답으로 간주됩니다."
    )
    fig2, ax2 = plt.subplots(figsize=(10, 1.6))
    # draw audio waveform
    librosa.display.waveshow(y, sr=sr, alpha=0.45, ax=ax2)
    # plot detected onsets (학생)
    ax2.vlines(detected_onsets, -1, 1, color='tab:blue', alpha=0.9, label='학생 온셋 (감지)')
    # plot reference onsets (정답)
    if len(ref_onsets) > 0:
        ax2.vlines(ref_onsets, -1, 1, color='tab:green', alpha=0.9, linestyle='dashed',
                   label='정답 온셋 (MIDI/기준)')
    ax2.legend(loc='upper right', ncol=2, fontsize='small')
    ax2.set_xlim(0, len(y) / sr)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_yticks([])
    ax2.set_xlabel("시간 (s)")
    st.pyplot(fig2)

    st.success("분석 완료")
    st.caption("참고: 그래프의 색상/표식은 위 설명을 따릅니다. 소음/다중 성부가 있으면 자동 추출 기준과 차이가 클 수 있습니다.")
