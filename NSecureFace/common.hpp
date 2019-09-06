//
//  common.hpp
//  NSecureFace
//
//  Created by Jun Liu on 2019/8/22.
//

#ifndef common_h
#define common_h

#include "nsecureface.h"
#include "json.hpp"

namespace nsecureface {
    
    void fatal(const char *func, int rv)
    {
        fprintf(stderr, "%s: %s\n", func, nng_strerror(rv));
        exit(1);
    }

    void Debug(NSecureFaceConfig& config)
    {
        printf("%s\n", std::string(110, '#').c_str());
        printf("%-30s: %d\n", "Device", config.device);
        printf("%-30s: %s\n", "Face Images", config.face_images.c_str());
        printf("%-30s: %s\n", "Embeddings", config.face_embeddings.c_str());
        printf("%-30s: %s\n", "Recognizer", config.face_recognizer.c_str());
        printf("%-30s: %s\n", "Labels", config.face_labels.c_str());
        printf("%-30s: %s\n", "DNN Network", config.dnn_network.c_str());
        printf("%-30s: %s\n", "DNN Weights", config.dnn_weights.c_str());
        printf("%-30s: %s\n", "Facemark Model", config.facemark_detector.c_str());
        printf("%-30s: %s\n", "Detector Directory", config.face_model.c_str());
        printf("%-30s: %s\n", "Face Recognition Service", config.face_recognition_service.c_str());
        printf("%-30s: %d\n", "Face Recognition Service Port", config.face_recognition_service_port);
        printf("%-30s: %s\n", "Image Server Url", config.image_server_url.c_str());
		printf("%-30s: %s\n", "Authentication Service Url", config.auth_service_url.c_str());
		printf("%-30s: %d\n", "Authentication Service Port", config.auth_service_port);
        printf("%s\n", std::string(110, '#').c_str());
    }
    
    NSecureFaceConfig LoadJsonConfig(std::string config_file_path)
    {
        using json = nlohmann::json;
        
        NSecureFaceConfig config;
        
        std::ifstream config_reader(config_file_path);
        json json_config;
        
        if (config_reader.is_open())
        {
            config_reader >> json_config;
            config_reader.close();
            
            config.device                        = json_config["device"].get<int>();
            config.face_images                   = json_config["face_images"].get<std::string>();
            config.face_embeddings               = json_config["face_embeddings"].get<std::string>();
            config.face_recognizer               = json_config["face_recognizer"].get<std::string>();
            config.face_labels                   = json_config["face_labels"].get<std::string>();
            config.face_model                    = json_config["face_model"].get<std::string>();
            config.dnn_network                   = json_config["face_detector"]["network"].get<std::string>();
            config.dnn_weights                   = json_config["face_detector"]["weights"].get<std::string>();
            config.facemark_detector             = json_config["facemark_detector"].get<std::string>();
            config.face_capture_unknown          = json_config["face_capture"]["face_unknown"].get<std::string>();
            config.face_capture_negative         = json_config["face_capture"]["face_negative"].get<std::string>();
            config.face_recognition_service      = json_config["face_recognition_service"]["host"].get<std::string>();
            config.face_recognition_service_port = json_config["face_recognition_service"]["port"].get<int>();
            config.image_server_url              = json_config["image_server_url"].get<std::string>();
			config.auth_service_url              = json_config["authentication_service"]["host"].get<std::string>();
			config.auth_service_port             = json_config["authentication_service"]["port"].get<int>();
			config.aws_region                    = json_config["aws"]["region"].get<std::string>();
        }
        else {
            printf("[ERROR] failed to open configuration file config.json");
        }
        
        return config;
    }
}

#endif /* common_h */
