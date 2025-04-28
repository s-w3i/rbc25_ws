#include "voice_record.h"

#include <ctime>
#include <iomanip>

SpeechProcess::SpeechProcess(const std::string& node_name)
    : Node(node_name)
{
    // Only store default ALSA params; do NOT open PCM
    params.channel = 1;  // mono
    params.format  = 2;  // S16_LE

    // Create the recording service
    record_service_ = this->create_service<std_srvs::srv::Trigger>(
        "record_audio",
        std::bind(&SpeechProcess::recordAudioCallback, this,
                  std::placeholders::_1, std::placeholders::_2)
    );

    RCLCPP_INFO(get_logger(), 
        "SpeechProcess node initialized. Service ready: '/record_audio'");
}

SpeechProcess::~SpeechProcess()
{
    // In case a recording is ongoing, stop it
    is_recording = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    RCLCPP_INFO(get_logger(), "SpeechProcess node shutdown");
}

// This function opens & configures the ALSA PCM device
// Only called once inside the service callback, ensures no pre-trigger capture
int SpeechProcess::record_params_init(record_handle_t* pcm_handle, const record_params_t* params)
{
    int err;
    snd_pcm_hw_params_t* hwparams = nullptr;

    // Desired parameters
    unsigned int rate = 16000;  // sample rate
    snd_pcm_uframes_t period_size = 512;
    snd_pcm_uframes_t buffer_size = period_size * 4; // example 4 periods

    // 1) Open PCM
    err = snd_pcm_open(&pcm_handle->pcm, RECORD_DEVICE_NAME,
                       SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "Cannot open audio device '%s': %s",
            RECORD_DEVICE_NAME, snd_strerror(err));
        return -1;
    }

    snd_pcm_hw_params_alloca(&hwparams);
    err = snd_pcm_hw_params_any(pcm_handle->pcm, hwparams);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_any() failed: %s", snd_strerror(err));
        goto error;
    }

    // Interleaved read mode
    err = snd_pcm_hw_params_set_access(pcm_handle->pcm, hwparams,
                                       SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_access() failed: %s", snd_strerror(err));
        goto error;
    }

    // Set PCM format
    pcm_handle->format = get_formattype_from_params(params);
    err = snd_pcm_hw_params_set_format(pcm_handle->pcm, hwparams, pcm_handle->format);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_format() failed: %s", snd_strerror(err));
        goto error;
    }

    // Set channels
    err = snd_pcm_hw_params_set_channels(pcm_handle->pcm, hwparams, params->channel);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_channels() failed: %s", snd_strerror(err));
        goto error;
    }

    // Set sample rate (16000)
    err = snd_pcm_hw_params_set_rate_near(pcm_handle->pcm, hwparams, &rate, nullptr);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_rate_near() failed: %s", snd_strerror(err));
        goto error;
    }
    pcm_handle->rate = rate;

    // Set period size
    err = snd_pcm_hw_params_set_period_size_near(pcm_handle->pcm, hwparams, &period_size, nullptr);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_period_size_near() failed: %s", snd_strerror(err));
        goto error;
    }

    // Set buffer size
    err = snd_pcm_hw_params_set_buffer_size_near(pcm_handle->pcm, hwparams, &buffer_size);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params_set_buffer_size_near() failed: %s", snd_strerror(err));
        goto error;
    }

    // Apply parameters
    err = snd_pcm_hw_params(pcm_handle->pcm, hwparams);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_hw_params() failed: %s", snd_strerror(err));
        goto error;
    }

    // Prepare PCM
    err = snd_pcm_prepare(pcm_handle->pcm);
    if (err < 0) {
        RCLCPP_ERROR(get_logger(),
            "snd_pcm_prepare() failed: %s", snd_strerror(err));
        goto error;
    }

    // Query final chunk_size
    snd_pcm_hw_params_get_period_size(hwparams, &period_size, nullptr);

    // Fill in local fields
    pcm_handle->chunk_size      = (size_t)period_size;
    pcm_handle->bits_per_sample = snd_pcm_format_width(pcm_handle->format) / 8; // e.g. 2
    pcm_handle->bits_per_frame  = pcm_handle->bits_per_sample * params->channel;
    pcm_handle->chunk_bytes     = pcm_handle->chunk_size * pcm_handle->bits_per_frame;

    // Allocate buffer for captured frames
    pcm_handle->buffer = (unsigned char*)malloc(pcm_handle->chunk_bytes);
    if (!pcm_handle->buffer) {
        RCLCPP_ERROR(get_logger(), "Cannot allocate capture buffer");
        goto error;
    }

    RCLCPP_INFO(get_logger(),
        "ALSA device opened: rate=%u, period_size=%lu, bits_per_sample=%zu",
        pcm_handle->rate,
        (unsigned long)pcm_handle->chunk_size,
        pcm_handle->bits_per_sample);

    return 0;

error:
    // Make sure we close if anything went wrong
    if (pcm_handle->pcm) {
        snd_pcm_close(pcm_handle->pcm);
        pcm_handle->pcm = nullptr;
    }
    return -1;
}

// Close PCM device and free buffer
void SpeechProcess::close_pcm(record_handle_t& pcm_handle)
{
    if (pcm_handle.buffer) {
        free(pcm_handle.buffer);
        pcm_handle.buffer = nullptr;
    }
    if (pcm_handle.pcm) {
        snd_pcm_close(pcm_handle.pcm);
        pcm_handle.pcm = nullptr;
    }
}

snd_pcm_format_t SpeechProcess::get_formattype_from_params(const record_params_t* params)
{
    switch (params->format) {
        case 0: return SND_PCM_FORMAT_S8;
        case 1: return SND_PCM_FORMAT_U8;
        case 2: return SND_PCM_FORMAT_S16_LE;  // typical default
        case 3: return SND_PCM_FORMAT_S16_BE;
        // add more if you need them
        default:
            return SND_PCM_FORMAT_S16_LE;
    }
}

bool SpeechProcess::check_silence(const unsigned char* buffer, size_t samples)
{
    float rms = 0.0f;
    const int16_t* audio_data = reinterpret_cast<const int16_t*>(buffer);

    // Compute sum of squares
    for (size_t i = 0; i < samples; ++i) {
        float val = static_cast<float>(audio_data[i]);
        rms += val * val;
    }
    // RMS and normalize to [-1,1]
    rms = std::sqrt(rms / samples) / 32768.0f;

    // If below threshold, consider silent
    return (rms < SILENCE_THRESHOLD);
}

/**
 * Write a 44-byte WAV header using the sample rate, channels, etc. from `rec` and `params`.
 */
void SpeechProcess::write_wav_header(
    FILE* file,
    uint32_t data_size,
    const record_handle_t& rec,
    const record_params_t& params)
{
    WavHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt,  "fmt ", 4);
    memcpy(header.data, "data", 4);

    header.fmt_size        = 16;      // PCM
    header.format          = 1;       // 1 => PCM
    header.channels        = params.channel;
    header.sample_rate     = rec.rate;
    header.bits_per_sample = rec.bits_per_sample * 8; // e.g. 16
    header.byte_rate       = header.sample_rate
                             * header.channels
                             * (header.bits_per_sample / 8);
    header.block_align     = header.channels
                             * (header.bits_per_sample / 8);

    header.data_size  = data_size;
    header.file_size  = data_size + sizeof(WavHeader) - 8; // minus riff+file_size

    fwrite(&header, 1, sizeof(header), file);
}

/**
 * One-shot service callback that:
 *  1) Opens & configures ALSA (no pre-service capture).
 *  2) Records until speech ends + 2s of silence or hits 10s max.
 *  3) Writes/updates WAV file header, closes ALSA device.
 */
void SpeechProcess::recordAudioCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    // If already recording, reject
    if (is_recording.exchange(true)) {
        response->success = false;
        response->message = "A recording is already in progress.";
        return;
    }

    // Setup a local record handle
    record_handle_t local_record;
    local_record.pcm    = nullptr;
    local_record.buffer = nullptr;

    // 1) Open ALSA device
    int ret = record_params_init(&local_record, &params);
    if (ret != 0) {
        response->success = false;
        response->message = "Failed to initialize ALSA device.";
        is_recording = false;
        return;
    }

    // 2) Create output directory
    const std::string save_dir = "/home/usern/rbc25_ws/audio";
    std::error_code ec;
    std::filesystem::create_directories(save_dir, ec);
    if (ec) {
        RCLCPP_ERROR(get_logger(),
                     "Failed to create directory '%s': %s",
                     save_dir.c_str(), ec.message().c_str());
        close_pcm(local_record);
        response->success = false;
        response->message = "Failed to create audio directory.";
        is_recording = false;
        return;
    }

    // 3) Build filename
    auto now       = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);

    char base_name[64];
    std::strftime(base_name, sizeof(base_name), "recording_%Y%m%d_%H%M%S",
                  std::localtime(&tt));

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%s_%04d.wav",
             save_dir.c_str(), base_name, file_counter.load());

    // 4) Open output WAV file
    FILE* wav_file = fopen(filename, "wb");
    if (!wav_file) {
        RCLCPP_ERROR(get_logger(), "Failed to create file '%s'", filename);
        close_pcm(local_record);
        response->success = false;
        response->message = "Failed to open WAV file.";
        is_recording = false;
        return;
    }

    // Write placeholder header (will update later)
    write_wav_header(wav_file, 0, local_record, params);
    uint32_t total_bytes = 0;

    RCLCPP_INFO(get_logger(), "Recording started: %s", filename);

    // Silence detection logic
    bool started_speech     = false;
    bool is_entirely_silent = true;
    float silent_seconds    = 0.0f;

    auto start_time = std::chrono::steady_clock::now();

    // 5) Capture loop
    while (is_recording) {
        // Read up to local_record.chunk_size frames
        int frames_read = snd_pcm_readi(local_record.pcm,
                                        local_record.buffer,
                                        local_record.chunk_size);
        if (frames_read < 0) {
            RCLCPP_WARN(get_logger(),
                        "snd_pcm_readi returned %d (%s)",
                        frames_read, snd_strerror(frames_read));
            break;
        } else if (frames_read == 0) {
            // no frames => continue
            continue;
        }

        // chunk duration in seconds
        float chunk_duration = static_cast<float>(frames_read) / local_record.rate;

        // Check if this chunk is silent
        bool current_silent = check_silence(local_record.buffer,
                                           frames_read * params.channel);

        if (!started_speech) {
            // Haven't heard speech yet
            if (!current_silent) {
                // We got a non-silent chunk => user started talking
                started_speech = true;
                is_entirely_silent = false;
                RCLCPP_INFO(get_logger(),
                            "Detected first speech - enabling silence timer");
            }
        } else {
            // We have started speech; watch for trailing silence
            if (current_silent) {
                silent_seconds += chunk_duration;
                if (silent_seconds >= SILENCE_TIMEOUT) {
                    RCLCPP_INFO(get_logger(),
                                "%.2f s of silence after speech - stopping",
                                silent_seconds);
                    break;
                }
            } else {
                // Non-silent frames => reset silence counter
                silent_seconds = 0.0f;
                is_entirely_silent = false;
            }
        }

        // Write captured frames to file
        size_t bytes_this_chunk = frames_read * local_record.bits_per_frame;
        fwrite(local_record.buffer, 1, bytes_this_chunk, wav_file);
        total_bytes += bytes_this_chunk;

        // Check for max record time
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= MAX_RECORD_SECONDS) {
            RCLCPP_INFO(get_logger(),
                        "Hit max record time (%d sec) - stopping",
                        MAX_RECORD_SECONDS);
            break;
        }
    }

    // 6) Done recording => update WAV header with correct data size
    fseek(wav_file, 0, SEEK_SET);
    write_wav_header(wav_file, total_bytes, local_record, params);
    fclose(wav_file);

    // Release is_recording
    is_recording = false;

    // 7) Close ALSA => no further capture
    close_pcm(local_record);

    // If entire file is silent, delete it
    if (is_entirely_silent) {
        remove(filename);
        RCLCPP_INFO(get_logger(),
                    "Only silence captured. Deleted file: %s",
                    filename);
        response->success = true;
        response->message = "File contained only silence, so it was deleted.";
        return;
    }

    // Otherwise, success
    file_counter++;
    RCLCPP_INFO(get_logger(),
                "Saved valid recording: %s", filename);

    response->success = true;
    response->message = std::string("Saved recording: ") + filename;
}

// Standard ROS2 main
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<SpeechProcess>("voice_control");
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
