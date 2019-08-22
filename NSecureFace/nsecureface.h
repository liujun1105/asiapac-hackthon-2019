#ifndef NSECUREFACE_SERVICE_H
#define NSECUREFACE_SERVICE_H

#include <string>

namespace nsecureface
{
    struct NSecureFaceConfig
    {       
        int         device;
        std::string face_images;
        std::string face_embeddings;
        std::string face_recognizer;
        std::string face_labels;
        std::string dnn_network;
        std::string dnn_weights;
        std::string face_model;
        std::string facemark_detector;
        std::string face_capture_unknown;
        std::string face_capture_negative;
        std::string face_recognition_service;
        int         face_recognition_service_port;
        std::string image_server_url;
    };
}

#endif
