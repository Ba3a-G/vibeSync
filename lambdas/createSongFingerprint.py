import os
import json
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import yt_dlp
from numpy import fft, array, maximum, log, hanning, abs
from typing import Dict, List, Optional, Any, TypeVar, Generic
from enum import IntEnum
from copy import copy
from typing import Dict, List, TypedDict, ClassVar, Final
from base64 import b64decode, b64encode
from math import  exp, sqrt
from binascii import crc32
from enum import IntEnum
from io import BytesIO
from ctypes import *
from dataclasses import dataclass, field

# signatuteFormat.py
# DATA_URI_PREFIX: Final[str] = 'data:audio/vnd.shazam.sig;base64,'

class SampleRate(IntEnum):  # Enum keys are sample rates in Hz
    _8000 = 1
    _11025 = 2
    _16000 = 3
    _32000 = 4
    _44100 = 5
    _48000 = 6


class FrequencyBand(IntEnum):  # Enum keys are frequency ranges in Hz
    _0_250 = -1  # Nothing above 250 Hz is actually stored
    _250_520 = 0
    _520_1450 = 1
    _1450_3500 = 2
    _3500_5500 = 3  # 3.5 KHz - 5.5 KHz should not be used in legacy mode


class RawSignatureHeader(LittleEndianStructure):
    _pack_ = True

    _fields_ = [
        ('magic1', c_uint32),
        ('crc32', c_uint32),  # CRC-32 for all following data (excluding first 8 bytes)
        ('size_minus_header', c_uint32),  # Total size minus header (48 bytes)
        ('magic2', c_uint32),  # Fixed 0x94119c00 - 00 9c 11 94
        ('void1', c_uint32 * 3),  # Void
        ('shifted_sample_rate_id', c_uint32),  # SampleRate left-shifted by 27
        ('void2', c_uint32 * 2),  # Void, or for "rolling window" mode
        ('number_samples_plus_divided_sample_rate', c_uint32),  # samples + sample_rate * 0.24
        ('fixed_value', c_uint32)  # Calculated as ((15 << 19) + 0x40000)
    ]


class PeakEncodingConstants:
    """Constants used for frequency peak encoding/decoding."""
    FFT_PASS_MARKER: Final[int] = 0xff
    FFT_PASS_LENGTH: Final[int] = 4
    PEAK_MAGNITUDE_BYTES: Final[int] = 2
    PEAK_FREQUENCY_BYTES: Final[int] = 2
    PEAK_RECORD_LENGTH: Final[int] = 5  # 1 + 2 + 2


@dataclass
class FrequencyPeak:
    """A single frequency peak detected in the audio."""
    __slots__ = ('fft_pass_number', 'peak_magnitude', 'corrected_peak_frequency_bin', 'sample_rate_hz')

    fft_pass_number: int
    peak_magnitude: int
    corrected_peak_frequency_bin: int
    sample_rate_hz: int

    def get_frequency_hz(self) -> float:
        """Convert FFT bin to frequency in Hz."""
        return self.corrected_peak_frequency_bin * (self.sample_rate_hz / 2 / 1024 / 64)

    def get_amplitude_pcm(self) -> float:
        """Calculate amplitude in PCM."""
        return sqrt(exp((self.peak_magnitude - 6144) / 1477.3) * (1 << 17) / 2) / 1024

    def get_seconds(self) -> float:
        """Calculate time position in seconds."""
        return (self.fft_pass_number * 128) / self.sample_rate_hz


class FrequencyPeakJSON(TypedDict):
    """JSON representation of a frequency peak."""
    fft_pass_number: int
    peak_magnitude: int
    corrected_peak_frequency_bin: int
    _frequency_hz: float
    _amplitude_pcm: float
    _seconds: float


class DecodedMessageJSON(TypedDict):
    """JSON representation of a decoded message."""
    sample_rate_hz: int
    number_samples: int
    _seconds: float
    frequency_band_to_peaks: Dict[str, List[FrequencyPeakJSON]]


@dataclass
class DecodedMessage:
    """Represents a decoded audio fingerprint signature."""
    sample_rate_hz: int = 0
    number_samples: int = 0
    frequency_band_to_sound_peaks: Dict[FrequencyBand, List[FrequencyPeak]] = field(default_factory=dict)

    # Class constants
    HEADER_SIZE: ClassVar[int] = 48
    MAGIC1: ClassVar[int] = 0xcafe2580
    MAGIC2: ClassVar[int] = 0x94119c00
    TLV_TYPE_FIXED: ClassVar[int] = 0x40000000
    BAND_ID_OFFSET: ClassVar[int] = 0x60030040

    @classmethod
    def decode_from_binary(cls, data: bytes) -> 'DecodedMessage':
        """Decode a binary signature into a DecodedMessage object."""
        self = cls()
        buf = BytesIO(data)

        # Extract checksummable data
        buf.seek(8)
        checksummable_data = buf.read()
        buf.seek(0)

        # Read header
        header = RawSignatureHeader()
        buf.readinto(header)

        # Validate header
        assert header.magic1 == cls.MAGIC1, "Invalid magic1 value"
        assert header.size_minus_header == len(data) - cls.HEADER_SIZE, "Invalid size in header"
        assert crc32(checksummable_data) & 0xffffffff == header.crc32, "CRC32 checksum mismatch"
        assert header.magic2 == cls.MAGIC2, "Invalid magic2 value"

        # Extract sample rate and number of samples
        self.sample_rate_hz = int(SampleRate(header.shifted_sample_rate_id >> 27).name.strip('_'))
        self.number_samples = int(header.number_samples_plus_divided_sample_rate - self.sample_rate_hz * 0.24)

        # Skip the fixed TLV section
        assert int.from_bytes(buf.read(4), 'little') == cls.TLV_TYPE_FIXED
        assert int.from_bytes(buf.read(4), 'little') == len(data) - cls.HEADER_SIZE

        # Process frequency bands and peaks
        self._decode_frequency_peaks(buf)

        return self

    def _decode_frequency_peaks(self, buf: BytesIO) -> None:
        """Process and decode frequency peaks from the buffer."""
        while True:
            tlv_header = buf.read(8)
            if not tlv_header:
                break

            frequency_band_id = int.from_bytes(tlv_header[:4], 'little')
            frequency_peaks_size = int.from_bytes(tlv_header[4:], 'little')

            # Calculate padding and prepare buffer for frequency peaks
            frequency_peaks_padding = -frequency_peaks_size % 4
            frequency_peaks_buf = BytesIO(buf.read(frequency_peaks_size))
            buf.read(frequency_peaks_padding)  # Skip padding

            # Decode the frequency band
            frequency_band = FrequencyBand(frequency_band_id - self.BAND_ID_OFFSET)
            self.frequency_band_to_sound_peaks[frequency_band] = self._decode_band_peaks(
                frequency_peaks_buf, frequency_band
            )

    def _decode_band_peaks(self, peaks_buf: BytesIO, band: FrequencyBand) -> List[FrequencyPeak]:
        """Decode all peaks for a specific frequency band."""
        peaks: List[FrequencyPeak] = []
        fft_pass_number = 0

        while True:
            raw_fft_pass = peaks_buf.read(1)
            if not raw_fft_pass:
                break

            fft_pass_offset = raw_fft_pass[0]

            # Check for FFT pass marker
            if fft_pass_offset == PeakEncodingConstants.FFT_PASS_MARKER:
                fft_pass_number = int.from_bytes(
                    peaks_buf.read(PeakEncodingConstants.FFT_PASS_LENGTH),
                    'little'
                )
                continue

            # Update the FFT pass number with the offset
            fft_pass_number += fft_pass_offset

            # Read peak data
            peak_magnitude = int.from_bytes(
                peaks_buf.read(PeakEncodingConstants.PEAK_MAGNITUDE_BYTES),
                'little'
            )
            corrected_peak_frequency_bin = int.from_bytes(
                peaks_buf.read(PeakEncodingConstants.PEAK_FREQUENCY_BYTES),
                'little'
            )

            # Add the peak to our collection
            peaks.append(
                FrequencyPeak(
                    fft_pass_number,
                    peak_magnitude,
                    corrected_peak_frequency_bin,
                    self.sample_rate_hz
                )
            )

        return peaks

    def encode_to_json(self) -> DecodedMessageJSON:
        """Encode the current object to a JSON format for debugging."""
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "number_samples": self.number_samples,
            "_seconds": self.number_samples / self.sample_rate_hz,
            "frequency_band_to_peaks": {
                frequency_band.name.strip('_'): [
                    {
                        "fft_pass_number": peak.fft_pass_number,
                        "peak_magnitude": peak.peak_magnitude,
                        "corrected_peak_frequency_bin": peak.corrected_peak_frequency_bin,
                        "_frequency_hz": peak.get_frequency_hz(),
                        "_amplitude_pcm": peak.get_amplitude_pcm(),
                        "_seconds": peak.get_seconds()
                    }
                    for peak in frequency_peaks
                ]
                for frequency_band, frequency_peaks in sorted(self.frequency_band_to_sound_peaks.items())
            }
        }

    def encode_to_binary(self) -> bytes:
        """Encode the current object to binary format."""
        # Create and initialize header
        header = RawSignatureHeader()
        header.magic1 = self.MAGIC1
        header.magic2 = self.MAGIC2
        header.shifted_sample_rate_id = int(getattr(SampleRate, f'_{self.sample_rate_hz}')) << 27
        header.fixed_value = ((15 << 19) + 0x40000)
        header.number_samples_plus_divided_sample_rate = int(self.number_samples + self.sample_rate_hz * 0.24)

        # Generate content buffer
        contents_buf = BytesIO()
        self._encode_frequency_peaks(contents_buf)
        content_bytes = contents_buf.getvalue()

        # Generate the full message
        buf = BytesIO()
        header.size_minus_header = len(content_bytes) + 8

        # Write header placeholder
        buf.write(bytes(header))

        # Write fixed TLV section
        buf.write(self.TLV_TYPE_FIXED.to_bytes(4, 'little'))
        buf.write((len(content_bytes) + 8).to_bytes(4, 'little'))

        # Write content
        buf.write(content_bytes)

        # Calculate and update CRC32
        buf.seek(8)
        header.crc32 = crc32(buf.read()) & 0xffffffff
        buf.seek(0)
        buf.write(bytes(header))

        return buf.getvalue()

    def _encode_frequency_peaks(self, contents_buf: BytesIO) -> None:
        """Encode all frequency peaks to the content buffer."""
        for frequency_band, frequency_peaks in sorted(self.frequency_band_to_sound_peaks.items()):
            if not frequency_peaks:
                continue

            peaks_buf = BytesIO()
            fft_pass_number = 0

            for peak in frequency_peaks:
                # Check if we need to insert a marker for a big jump in FFT pass numbers
                if peak.fft_pass_number - fft_pass_number >= PeakEncodingConstants.FFT_PASS_MARKER:
                    peaks_buf.write(bytes([PeakEncodingConstants.FFT_PASS_MARKER]))
                    peaks_buf.write(peak.fft_pass_number.to_bytes(
                        PeakEncodingConstants.FFT_PASS_LENGTH, 'little'
                    ))
                    fft_pass_number = peak.fft_pass_number

                # Write the peak data
                peaks_buf.write(bytes([peak.fft_pass_number - fft_pass_number]))
                peaks_buf.write(peak.peak_magnitude.to_bytes(
                    PeakEncodingConstants.PEAK_MAGNITUDE_BYTES, 'little'
                ))
                peaks_buf.write(peak.corrected_peak_frequency_bin.to_bytes(
                    PeakEncodingConstants.PEAK_FREQUENCY_BYTES, 'little'
                ))

                fft_pass_number = peak.fft_pass_number

            # Write the TLV header and data for this frequency band
            peaks_data = peaks_buf.getvalue()
            contents_buf.write((self.BAND_ID_OFFSET + int(frequency_band)).to_bytes(4, 'little'))
            contents_buf.write(len(peaks_data).to_bytes(4, 'little'))
            contents_buf.write(peaks_data)

            # Add padding to align to 4-byte boundary
            padding_bytes = -len(peaks_data) % 4
            if padding_bytes:
                contents_buf.write(b'\x00' * padding_bytes)



# Algo BS
# Pre-compute the Hanning matrix once (wipe trailing and leading zeros)
HANNING_MATRIX = hanning(2050)[1:-1]

T = TypeVar('T')


class RingBuffer(Generic[T]):
    """Efficient ring buffer implementation with type annotations."""

    def __init__(self, buffer_size: int, default_value: Optional[T] = None):
        self.buffer: List[T] = [copy(default_value) for _ in range(buffer_size)] if default_value is not None else [
                                                                                                                       None] * buffer_size  # type: ignore
        self.position: int = 0
        self.buffer_size: int = buffer_size
        self.num_written: int = 0

    def append(self, value: T) -> None:
        """Add a value to the ring buffer."""
        self.buffer[self.position] = value
        self.position = (self.position + 1) % self.buffer_size
        self.num_written += 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Handle slicing
            return [self.buffer[i % self.buffer_size] for i in range(
                index.start if index.start is not None else 0,
                index.stop if index.stop is not None else self.buffer_size,
                index.step if index.step is not None else 1
            )]
        # Handle integer indexing
        return self.buffer[index % self.buffer_size]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            # Handle slice assignment
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else self.buffer_size
            step = index.step if index.step is not None else 1

            for i, v in zip(range(start, stop, step), value):
                self.buffer[i % self.buffer_size] = v
        else:
            # Handle integer indexing
            self.buffer[index % self.buffer_size] = value


class SignatureGenerator:
    def __init__(self):
        # Configuration
        self.MAX_TIME_SECONDS = 30.0
        self.MAX_PEAKS = 1000
        self.SAMPLE_RATE = 16000

        # Processing state
        self.input_pending_processing: List[int] = []
        self.samples_processed: int = 0

        # Ring buffers
        self.ring_buffer_of_samples: RingBuffer[int] = RingBuffer(buffer_size=2048, default_value=0)
        self.fft_outputs: RingBuffer[List[float]] = RingBuffer(buffer_size=256, default_value=[0.0] * 1025)
        self.spread_ffts_output: RingBuffer[List[float]] = RingBuffer(buffer_size=256, default_value=[0] * 1025)

        # Initialize signature object
        self.reset_signature()

    def reset_signature(self) -> None:
        """Reset the signature object and buffers for the next processing cycle."""
        self.next_signature = DecodedMessage()
        self.next_signature.sample_rate_hz = self.SAMPLE_RATE
        self.next_signature.number_samples = 0
        self.next_signature.frequency_band_to_sound_peaks = {}

        self.ring_buffer_of_samples = RingBuffer(buffer_size=2048, default_value=0)
        self.fft_outputs = RingBuffer(buffer_size=256, default_value=[0.0] * 1025)
        self.spread_ffts_output = RingBuffer(buffer_size=256, default_value=[0] * 1025)

    def feed_input(self, s16le_mono_samples: List[int]) -> None:
        """Add signed 16-bit 16 KHz mono PCM samples for signature generation."""
        self.input_pending_processing.extend(s16le_mono_samples)

    def get_next_signature(self) -> Optional[DecodedMessage]:
        """
        Process pending input samples and return a signature when enough data is gathered.
        Returns None if no more samples to process.
        """
        if len(self.input_pending_processing) - self.samples_processed < 128:
            return None

        # Process available samples until we reach time/peak limits
        while (len(self.input_pending_processing) - self.samples_processed >= 128 and
               (self.next_signature.number_samples / self.next_signature.sample_rate_hz < self.MAX_TIME_SECONDS or
                sum(len(peaks) for peaks in
                    self.next_signature.frequency_band_to_sound_peaks.values()) < self.MAX_PEAKS)):
            chunk = self.input_pending_processing[self.samples_processed:self.samples_processed + 128]
            self.process_input(chunk)
            self.samples_processed += 128

        # Return the completed signature and reset for next one
        returned_signature = self.next_signature
        self.reset_signature()
        return returned_signature

    def process_input(self, s16le_mono_samples: List[int]) -> None:
        """Process a batch of audio samples to extract features."""
        self.next_signature.number_samples += len(s16le_mono_samples)

        for position in range(0, len(s16le_mono_samples), 128):
            chunk = s16le_mono_samples[position:position + 128]
            self.do_fft(chunk)
            self.do_peak_spreading_and_recognition()

    def do_fft(self, batch_of_128_s16le_mono_samples: List[int]) -> None:
        """Perform Fast Fourier Transform on the audio samples."""
        # Update ring buffer with new samples
        end_pos = (self.ring_buffer_of_samples.position + len(
            batch_of_128_s16le_mono_samples)) % self.ring_buffer_of_samples.buffer_size

        if end_pos > self.ring_buffer_of_samples.position:
            # Contiguous segment
            self.ring_buffer_of_samples.buffer[
            self.ring_buffer_of_samples.position:end_pos] = batch_of_128_s16le_mono_samples
        else:
            # Wrapping around the buffer
            first_part = self.ring_buffer_of_samples.buffer_size - self.ring_buffer_of_samples.position
            self.ring_buffer_of_samples.buffer[self.ring_buffer_of_samples.position:] = batch_of_128_s16le_mono_samples[
                                                                                        :first_part]
            self.ring_buffer_of_samples.buffer[:end_pos] = batch_of_128_s16le_mono_samples[first_part:]

        self.ring_buffer_of_samples.position = end_pos
        self.ring_buffer_of_samples.num_written += len(batch_of_128_s16le_mono_samples)

        # Extract data from ring buffer
        excerpt_from_ring_buffer = (
                self.ring_buffer_of_samples[self.ring_buffer_of_samples.position:] +
                self.ring_buffer_of_samples[:self.ring_buffer_of_samples.position]
        )

        # Apply Hanning window and perform FFT
        fft_results = fft.rfft(HANNING_MATRIX * excerpt_from_ring_buffer)

        # Calculate magnitude
        fft_results = (fft_results.real ** 2 + fft_results.imag ** 2) / (1 << 17)
        fft_results = maximum(fft_results, 1e-10)  # Avoid very small values

        self.fft_outputs.append(fft_results)

    def do_peak_spreading_and_recognition(self) -> None:
        """Analyze FFT outputs to detect and spread peaks."""
        self.do_peak_spreading()

        if self.spread_ffts_output.num_written >= 46:
            self.do_peak_recognition()

    def do_peak_spreading(self) -> None:
        """Perform frequency-domain and time-domain spreading of peak values."""
        origin_last_fft = self.fft_outputs[self.fft_outputs.position - 1]
        spread_last_fft = list(origin_last_fft)

        # Frequency-domain spreading
        for position in range(1023):
            spread_last_fft[position] = max(spread_last_fft[position:position + 3])

        # Time-domain spreading
        for position in range(1025):
            max_value = spread_last_fft[position]

            for former_fft_num in [-1, -3, -6]:
                idx = (self.spread_ffts_output.position + former_fft_num) % self.spread_ffts_output.buffer_size
                former_fft_output = self.spread_ffts_output[idx]
                former_fft_output[position] = max_value = max(former_fft_output[position], max_value)

        self.spread_ffts_output.append(spread_last_fft)

    def do_peak_recognition(self) -> None:
        """Identify significant frequency peaks from the spread FFT outputs."""
        fft_minus_46 = self.fft_outputs[(self.fft_outputs.position - 46) % self.fft_outputs.buffer_size]
        fft_minus_49 = self.spread_ffts_output[
            (self.spread_ffts_output.position - 49) % self.spread_ffts_output.buffer_size]
        fft_minus_53 = self.spread_ffts_output[
            (self.spread_ffts_output.position - 53) % self.spread_ffts_output.buffer_size]
        fft_minus_45 = self.spread_ffts_output[
            (self.spread_ffts_output.position - 45) % self.spread_ffts_output.buffer_size]

        # Define frequency bands
        FREQ_BANDS = {
            (250, 520): FrequencyBand._250_520,
            (520, 1450): FrequencyBand._520_1450,
            (1450, 3500): FrequencyBand._1450_3500,
            (3500, 5500): FrequencyBand._3500_5500
        }

        for bin_position in range(10, 1015):
            # Check if bin is large enough to be a peak
            if fft_minus_46[bin_position] < 1 / 64 or fft_minus_46[bin_position] < fft_minus_49[bin_position - 1]:
                continue

            # Check frequency-domain local minimum
            neighbor_offsets = [*range(-10, -3, 3), -3, 1, *range(2, 9, 3)]
            max_neighbor = max(fft_minus_49[bin_position + offset] for offset in neighbor_offsets)

            if fft_minus_46[bin_position] <= max_neighbor:
                continue

            # Check time-domain local minimum
            other_offsets = [-53, -45, *range(165, 201, 7), *range(214, 250, 7)]
            max_other_neighbor = max(
                self.spread_ffts_output[(self.spread_ffts_output.position + offset) %
                                        self.spread_ffts_output.buffer_size][bin_position - 1]
                for offset in other_offsets
            )

            if fft_minus_46[bin_position] <= max_other_neighbor:
                continue

            # This is a peak - calculate its properties
            fft_number = self.spread_ffts_output.num_written - 46

            # Calculate peak magnitude and correction
            peak_magnitude = log(max(1 / 64, fft_minus_46[bin_position])) * 1477.3 + 6144
            peak_magnitude_before = log(max(1 / 64, fft_minus_46[bin_position - 1])) * 1477.3 + 6144
            peak_magnitude_after = log(max(1 / 64, fft_minus_46[bin_position + 1])) * 1477.3 + 6144

            peak_variation_1 = peak_magnitude * 2 - peak_magnitude_before - peak_magnitude_after

            if peak_variation_1 <= 0:
                continue

            peak_variation_2 = (peak_magnitude_after - peak_magnitude_before) * 32 / peak_variation_1
            corrected_peak_frequency_bin = bin_position * 64 + peak_variation_2

            # Determine frequency band
            frequency_hz = corrected_peak_frequency_bin * (self.SAMPLE_RATE / 2 / 1024 / 64)

            if frequency_hz < 250:
                continue

            # Assign to appropriate frequency band
            band = None
            for (low, high), freq_band in FREQ_BANDS.items():
                if low <= frequency_hz < high:
                    band = freq_band
                    break

            if band is None:
                continue

            # Store the peak
            if band not in self.next_signature.frequency_band_to_sound_peaks:
                self.next_signature.frequency_band_to_sound_peaks[band] = []

            self.next_signature.frequency_band_to_sound_peaks[band].append(
                FrequencyPeak(
                    fft_number,
                    int(peak_magnitude),
                    int(corrected_peak_frequency_bin),
                    self.SAMPLE_RATE
                )
            )

# gen signature
def create_fingerprint(samples: tuple[List[int],int]) -> DecodedMessage:
    # """Create fingerprint from an audio file path."""
    # print(f"Processing {os.path.basename(audio_file_path)}...")

    # # Convert audio file to raw samples
    # samples, original_sr = convert_audio_to_raw_samples(audio_file_path)

    # Create signature generator
    generator = SignatureGenerator()

    # Feed samples to generator
    generator.feed_input(samples)

    # Get signature
    signature = generator.get_next_signature()

    if signature is None:
        raise ValueError("Failed to generate fingerprint")

    return signature

def lambda_handler(event, context):
    """
    AWS Lambda function that downloads the first 30 seconds of a YouTube video,
    converts it to 16-bit mono 16kHz PCM format, and saves it to the temp directory.
    
    Parameters:
    event (dict): Should contain a 'youtube_url' key with the YouTube video URL
    context (LambdaContext): Lambda context object
    
    Returns:
    dict: Response with status and file path information
    """
    try:
        if 'youtube_url' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing youtube_url parameter'})
            }
        
        youtube_url = event['youtube_url']
        
        if 'tmp_dir' not in event:
            temp_dir = tempfile.gettempdir()
        else:
            temp_dir = event['tmp_dir']
        print(f"Using temp directory: {temp_dir}")
        
        video_id = youtube_url.split('v=')[-1].split('&')[0]
        
        ydl_opts = {
            'quiet': False,
            'no_warnings': False,
            'format': 'worstaudio/worst',
            'skip_download': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id', video_id)
        
        # Now download using ffmpeg with the time limit
        temp_audio_file = os.path.join(temp_dir, f"{video_id}_original.wav")

        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': os.path.join(temp_dir, f"{video_id}_original.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': {
                'ffmpeg': ['-t', '30']
            },
            'prefer_ffmpeg': True,
            'keepvideo': False
        }
        
        print(f"Downloading audio from {youtube_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        downloaded_file = None
        for file in os.listdir(temp_dir):
            if file.startswith(f"{video_id}_original") and file.endswith('.wav'):
                downloaded_file = os.path.join(temp_dir, file)
                break
        
        if not downloaded_file:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Could not find downloaded file'})
            }
        
        print(f"Downloaded file: {downloaded_file}")
        
        print("Loading with librosa and downsampling...")
        y, _ = librosa.load(downloaded_file, sr=16000, mono=True)
        
        output_file = os.path.join(temp_dir, f"{video_id}_fingerprint.wav")

        print(create_fingerprint(y))

        sf.write(output_file, y, 16000, subtype='PCM_16')
        
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Audio processed successfully',
                'video_id': video_id,
                'output_file': output_file,
                'duration': len(y) / 16000
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

if __name__ == '__main__':
    event = {'youtube_url': 'https://www.youtube.com/watch?v=0RDI9CMilhk', 'tmp_dir': './tmp'}
    context = None
    response = lambda_handler(event, context)
    print(response)