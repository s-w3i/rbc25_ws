#ifndef VOICE_RECORD_H
#define VOICE_RECORD_H

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <alsa/asoundlib.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

// ALSA device name
#define RECORD_DEVICE_NAME "default"

// User-defined recording parameters (channels, ALSA format code)
typedef struct {
    int channel;  // e.g. 1 => mono, 2 => stereo
    int format;   // e.g. 2 => SND_PCM_FORMAT_S16_LE
} record_params_t;

// ALSA PCM handle & metadata
typedef struct {
    snd_pcm_t* pcm;
    snd_pcm_format_t format;
    unsigned int rate;
    size_t chunk_size;
    size_t bits_per_sample;
    size_t bits_per_frame;
    size_t chunk_bytes;
    unsigned char* buffer;
} record_handle_t;

// Minimal WAV header struct for 16-bit PCM
typedef struct {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // file size minus 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // 16 for PCM
    uint16_t format;        // 1 => PCM
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];           // "data"
    uint32_t data_size;     // size of audio data
} WavHeader;

class SpeechProcess : public rclcpp::Node {
public:
    SpeechProcess(const std::string& node_name);
    ~SpeechProcess();

private:
    // We do NOT open ALSA in the constructor, so we only have a default 'params'.
    record_params_t params;

    // Flag controlling if a recording is in progress
    std::atomic<bool> is_recording{false};
    // Incremented each time a valid file is saved
    std::atomic<int> file_counter{0};

    // ROS service for triggering a single recording session
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr record_service_;

    // Configuration constants
    static constexpr float SILENCE_THRESHOLD  = 0.03f; // RMS threshold for silence
    static constexpr float SILENCE_TIMEOUT    = 2.0f;  // Stop 2s after speech ends
    static constexpr int   MAX_RECORD_SECONDS = 10;    // Hard stop at 10s

    // Internal functions
    int record_params_init(record_handle_t* pcm_handle, const record_params_t* params);
    static snd_pcm_format_t get_formattype_from_params(const record_params_t* params);
    void close_pcm(record_handle_t& pcm_handle);

    bool check_silence(const unsigned char* buffer, size_t samples);

    // **Important**: Updated to accept the local record & params
    void write_wav_header(FILE* file,
                          uint32_t data_size,
                          const record_handle_t& rec,
                          const record_params_t& params);

    // Service callback that captures one WAV file
    void recordAudioCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response
    );
};

#endif // VOICE_RECORD_H
