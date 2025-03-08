#include "voice_record.h"
#include <filesystem>

SpeechProcess::SpeechProcess(const std::string& node_name) 
    : rclcpp::Node(node_name) {
    // Initialize ALSA parameters
    params.channel = 1;
    params.format = 2;  // S16_LE
    init_success = record_params_init(&record, &params);

    // Setup wake-up subscription
    awake_sub_ = create_subscription<std_msgs::msg::Int8>(
        "awake_flag", 10,
        [this](const std_msgs::msg::Int8::SharedPtr msg) {
            if(msg->data == 1 && !active_mode) {
                active_mode = true;
                silent_count = 0;
                file_counter = 0;
                RCLCPP_INFO(get_logger(), "System awakened - starting continuous recording");
                if(!is_recording) {
                    start_recording();
                }
            }
        });
}

SpeechProcess::~SpeechProcess() {
    finish_record_sound();
    RCLCPP_INFO(get_logger(), "Voice control node shutdown");
}

void SpeechProcess::write_wav_header(FILE* file, uint32_t data_size) {
    WavHeader header;
    memcpy(header.riff, "RIFF", 4);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    memcpy(header.data, "data", 4);
    
    header.fmt_size = 16;
    header.format = 1;
    header.channels = params.channel;
    header.sample_rate = record.rate;
    header.bits_per_sample = record.bits_per_sample * 8;
    header.byte_rate = record.rate * params.channel * record.bits_per_sample;
    header.block_align = params.channel * record.bits_per_sample;
    header.data_size = data_size;
    header.file_size = data_size + sizeof(WavHeader) - 8;
    
    fwrite(&header, 1, sizeof(header), file);
}

void SpeechProcess::start_recording()
{
    if(is_recording.exchange(true)) return;

    std::thread([this]() {
        // Create output directory
        const std::string save_dir = "/home/robot11/rbc25_ws/audio";
        std::error_code ec;
        std::filesystem::create_directories(save_dir, ec);
        if(ec) {
            RCLCPP_ERROR(get_logger(), "Failed to create directory: %s", ec.message().c_str());
            is_recording = false;
            return;
        }

        bool is_silent = true;
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        char base_name[50];
        char filename[256]; // Increased buffer size for path
        
        std::strftime(base_name, sizeof(base_name), "recording_%Y%m%d_%H%M%S", std::localtime(&time));
        snprintf(filename, sizeof(filename), "%s/%s_%04d.wav", 
                save_dir.c_str(), base_name, file_counter.load());

        FILE* wav_file = fopen(filename, "wb");
        if(!wav_file) {
            RCLCPP_ERROR(get_logger(), "Failed to create file: %s", filename);
            is_recording = false;
            return;
        }

        write_wav_header(wav_file, 0);
        uint32_t total_bytes = 0;
        auto start_time = std::chrono::steady_clock::now();
        float silent_seconds = 0.0f;

        while(active_mode && is_recording) {
            int ret = snd_pcm_readi(record.pcm, record.buffer, record.chunk_size);
            if(ret <= 0) break;

            // Calculate silence duration
            bool current_silent = check_silence(record.buffer, ret * params.channel);
            float chunk_duration = static_cast<float>(ret) / record.rate;
            
            if(current_silent) {
                silent_seconds += chunk_duration;
                if(silent_seconds >= 2.0f) {
                    RCLCPP_INFO(get_logger(), "2 seconds of silence detected - stopping recording");
                    break;
                }
            } else {
                silent_seconds = 0.0f;  // Reset counter on any audio
                is_silent = false;
            }

            // Write data
            size_t bytes = ret * record.bits_per_frame;
            total_bytes += bytes;
            fwrite(record.buffer, 1, bytes, wav_file);

            // Check maximum duration
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            if(duration.count() >= MAX_RECORD_SECONDS) break;
        }

        // Finalize file
        fseek(wav_file, 0, SEEK_SET);
        write_wav_header(wav_file, total_bytes);
        fclose(wav_file);

        // Handle silent files
        if(is_silent) {
            remove(filename);
            if(++silent_count >= MAX_SILENT_FILES) {
                active_mode = false;
                RCLCPP_WARN(get_logger(), "10 consecutive silent files - system dormant");
            }
        } else {
            silent_count = 0;
            file_counter++;
            RCLCPP_INFO(get_logger(), "Saved valid recording: %s", filename);
        }

        is_recording = false;
        
        // Start next recording if still active
        if(active_mode) {
            start_recording();
        }
    }).detach();
}

bool SpeechProcess::check_silence(const unsigned char* buffer, size_t samples) {
    float rms = 0.0f;
    const int16_t* audio_data = reinterpret_cast<const int16_t*>(buffer);
    
    for(size_t i = 0; i < samples; ++i) {
        rms += audio_data[i] * audio_data[i];
    }
    rms = std::sqrt(rms / samples) / 32768.0f;
    return rms < SILENCE_THRESHOLD;
}

int SpeechProcess::record_params_init(record_handle_t* pcm_handle, record_params_t* params) {
    int err;
    snd_pcm_hw_params_t* hwparams;
    unsigned int rate = 16000;

    if((err = snd_pcm_open(&pcm_handle->pcm, RECORD_DEVICE_NAME, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot open audio device: %s", snd_strerror(err));
        return -1;
    }

    snd_pcm_hw_params_alloca(&hwparams);
    if((err = snd_pcm_hw_params_any(pcm_handle->pcm, hwparams)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot initialize parameters: %s", snd_strerror(err));
        goto error;
    }

    if((err = snd_pcm_hw_params_set_access(pcm_handle->pcm, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot set access type: %s", snd_strerror(err));
        goto error;
    }

    pcm_handle->format = get_formattype_from_params(params);
    if((err = snd_pcm_hw_params_set_format(pcm_handle->pcm, hwparams, pcm_handle->format)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot set format: %s", snd_strerror(err));
        goto error;
    }

    if((err = snd_pcm_hw_params_set_channels(pcm_handle->pcm, hwparams, params->channel)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot set channels: %s", snd_strerror(err));
        goto error;
    }

    if((err = snd_pcm_hw_params_set_rate_near(pcm_handle->pcm, hwparams, &rate, 0)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot set rate: %s", snd_strerror(err));
        goto error;
    }
    pcm_handle->rate = rate;

    if((err = snd_pcm_hw_params(pcm_handle->pcm, hwparams)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot set parameters: %s", snd_strerror(err));
        goto error;
    }

    if((err = snd_pcm_prepare(pcm_handle->pcm)) < 0) {
        RCLCPP_ERROR(get_logger(), "Cannot prepare interface: %s", snd_strerror(err));
        goto error;
    }

    pcm_handle->chunk_size = buffer_frames;
    pcm_handle->bits_per_sample = snd_pcm_format_width(pcm_handle->format) / 8;
    pcm_handle->bits_per_frame = pcm_handle->bits_per_sample * params->channel;
    pcm_handle->chunk_bytes = pcm_handle->chunk_size * pcm_handle->bits_per_frame;
    pcm_handle->buffer = (unsigned char*)malloc(pcm_handle->chunk_bytes);

    if(!pcm_handle->buffer) {
        RCLCPP_ERROR(get_logger(), "Cannot allocate audio buffer");
        goto error;
    }

    return 0;

error:
    snd_pcm_close(pcm_handle->pcm);
    return -1;
}

snd_pcm_format_t SpeechProcess::get_formattype_from_params(record_params_t* params) {
    switch(params->format) {
        case 0: return SND_PCM_FORMAT_S8;
        case 1: return SND_PCM_FORMAT_U8;
        case 2: return SND_PCM_FORMAT_S16_LE;
        case 3: return SND_PCM_FORMAT_S16_BE;
        default: return SND_PCM_FORMAT_S16_LE;
    }
}

int SpeechProcess::finish_record_sound() {
    active_mode = false;
    if(is_recording) {
        is_recording = false;
        int timeout = 0;
        while(timeout++ < 50 && is_recording) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    if(record.buffer) {
        free(record.buffer);
        record.buffer = nullptr;
    }
    if(!init_success && record.pcm) {
        snd_pcm_close(record.pcm);
        record.pcm = nullptr;
    }
    return 0;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SpeechProcess>("voice_control"));
    rclcpp::shutdown();
    return 0;
}