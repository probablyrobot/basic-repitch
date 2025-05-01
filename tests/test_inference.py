#!/usr/bin/env python
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import faulthandler
import os
import pathlib
import tempfile

import librosa
import numpy as np
import numpy.typing as npt
import pretty_midi

from basic_pitch import ICASSP_2022_MODEL_PATH, inference
from basic_pitch.constants import (
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
)

RESOURCES_PATH = pathlib.Path(__file__).parent / "resources"

faulthandler.enable()


def test_predict() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    model_output, midi_data, note_events = inference.predict(
        test_audio_path,
        inference.Model(ICASSP_2022_MODEL_PATH),
    )
    assert set(model_output.keys()) == {"note", "onset", "contour"}
    assert model_output["note"].shape == model_output["onset"].shape
    assert isinstance(midi_data, pretty_midi.PrettyMIDI)
    lowest_supported_midi = 21
    note_pitch_min = [n[2] >= lowest_supported_midi for n in note_events]
    note_pitch_max = [n[2] <= lowest_supported_midi + ANNOTATIONS_N_SEMITONES for n in note_events]
    assert all(note_pitch_min)
    assert all(note_pitch_max)
    assert isinstance(note_events, list)

    expected_model_output = np.load(RESOURCES_PATH / "vocadito_10" / "model_output.npz", allow_pickle=True)[
        "arr_0"
    ].item()
    for k in expected_model_output.keys():
        np.testing.assert_allclose(expected_model_output[k], model_output[k], atol=1e-4, rtol=0)

    expected_note_events = np.load(RESOURCES_PATH / "vocadito_10" / "note_events.npz", allow_pickle=True)["arr_0"]
    assert len(expected_note_events) == len(note_events)
    for expected, calculated in zip(expected_note_events, note_events, strict=False):
        for i in range(len(expected)):
            np.testing.assert_allclose(expected[i], calculated[i], atol=1e-4, rtol=0)


def test_predict_with_saves() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        inference.predict_and_save(
            [test_audio_path],
            tmpdir,
            True,
            True,
            True,
            True,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        expected_midi_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.mid")
        expected_csv_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.csv")
        expected_npz_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.npz")
        expected_sonif_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.wav")

        for output_path in [
            expected_midi_path,
            expected_csv_path,
            expected_npz_path,
            expected_sonif_path,
        ]:
            assert os.path.exists(output_path)


def test_predict_onset_threshold() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for onset_threshold in [0, 0.3, 0.8, 1]:
        inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
        )


def test_predict_frame_threshold() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for frame_threshold in [0, 0.3, 0.8, 1]:
        inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            frame_threshold=frame_threshold,
        )


def test_predict_min_note_length() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for minimum_note_length in [10, 100, 1000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            minimum_note_length=minimum_note_length,
        )
        min_len_s = minimum_note_length / 1000.0
        note_lengths = [n[1] - n[0] >= min_len_s for n in note_events]
        assert all(note_lengths)


def test_predict_min_freq() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for minimum_frequency in [40, 80, 200, 2000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            minimum_frequency=minimum_frequency,
        )
        min_freq_midi = np.round(librosa.hz_to_midi(minimum_frequency))
        note_pitch = [n[2] >= min_freq_midi for n in note_events]
        assert all(note_pitch)


def test_predict_max_freq() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for maximum_frequency in [40, 80, 200, 2000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            maximum_frequency=maximum_frequency,
        )
        max_freq_midi = np.round(librosa.hz_to_midi(maximum_frequency))
        note_pitch = [n[2] <= max_freq_midi for n in note_events]
        assert all(note_pitch)


def test_window_audio_file() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    audio, _ = librosa.load(str(test_audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    audio_windowed, window_times = zip(
        *inference.window_audio_file(audio, AUDIO_N_SAMPLES - 30 * FFT_HOP), strict=False
    )
    assert len(audio_windowed) == 6
    assert len(window_times) == 6
    for time in window_times:
        assert time["start"] <= time["end"]
    np.testing.assert_equal(audio[:AUDIO_N_SAMPLES], np.squeeze(audio_windowed[0]))


def test_get_audio_input() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    audio, _ = librosa.load(str(test_audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    overlap_len = 30 * FFT_HOP
    audio = np.concatenate([np.zeros((overlap_len // 2,), dtype=np.float32), audio])
    audio_windowed: list[npt.NDArray[np.float32]] = []
    window_times: list[dict[str, float]] = []
    audio_chunks = list(inference.get_audio_input(test_audio_path, overlap_len, AUDIO_N_SAMPLES - overlap_len))
    for audio_window, window_time, _ in audio_chunks:
        audio_windowed.append(audio_window)
        window_times.append(window_time)
    audio_windowed = np.array(audio_windowed)
    assert len(audio_windowed) == 6
    assert len(window_times) == 6
    for time in window_times:
        assert time["start"] <= time["end"]
    np.testing.assert_equal(audio[:AUDIO_N_SAMPLES], np.squeeze(audio_windowed[0]))

    assert audio_chunks[-1][2] == 200607


def test_inference():
    try:
        model = inference.Model("path/to/model")
        assert model.model_type in {"tensorflow", "tflite", "onnx", "coreml"}
    except Exception as e:
        raise AssertionError(f"Failed to create Model: {e}")


def test_process_audio_chunk():
    audio_chunk = np.random.random(1000)
    model = inference.Model("path/to/model")
    chunk_length = len(audio_chunk)
    result = inference._process_audio_chunk(audio_chunk, model, model.model_type, chunk_length)
    assert isinstance(result, dict)
    assert result["note"].shape[0] == chunk_length
