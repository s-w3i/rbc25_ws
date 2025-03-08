#ifndef VOICE_RECORD_H
#define VOICE_RECORD_H

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int8.hpp>
#include <alsa/asoundlib.h>
#include <atomic>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
#include <thread>
#include <cstdio>

#define RECORD_DEVICE_NAME "default"
#define buffer_frames 512

typedef struct {
    int channel;
    int format;
} record_params_t;

typedef struct {
    snd_pcm_t *pcm;
    snd_pcm_format_t format;
    unsigned int rate;
    size_t chunk_size;
    size_t bits_per_sample;
    size_t bits_per_frame;
    size_t chunk_bytes;
    unsigned char *buffer;
} record_handle_t;

typedef struct {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
} WavHeader;

class SpeechProcess : public rclcpp::Node {
public:
    SpeechProcess(const std::string& node_name);
    ~SpeechProcess();

private:
    // ALSA members
    record_handle_t record;
    record_params_t params;
    int init_success;

    // Recording control
    std::atomic<bool> active_mode{false};
    std::atomic<bool> is_recording{false};
    std::atomic<int> silent_count{0};
    std::atomic<int> file_counter{0};
    
    // Configuration
    static constexpr float SILENCE_THRESHOLD = 0.1f;
    static constexpr int MAX_SILENT_FILES = 10;
    static constexpr int SILENCE_TIMEOUT = 3;
    static constexpr int MAX_RECORD_SECONDS = 10;

    // ROS members
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr awake_sub_;

    // Core functions
    void start_recording();
    void write_wav_header(FILE* file, uint32_t data_size);
    int record_params_init(record_handle_t* pcm_handle, record_params_t* params);
    static snd_pcm_format_t get_formattype_from_params(record_params_t* params);
    int finish_record_sound();
    bool check_silence(const unsigned char* buffer, size_t samples);
};

#endif