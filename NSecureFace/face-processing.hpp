//
//  face-processing.hpp
//  NSecureFaceService
//
//  Created by Jun Liu on 2019/8/17.
//

#ifndef face_processing_hpp
#define face_processing_hpp

#include <stdio.h>
#include <string>
#include "nsecureface.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <unordered_map>

namespace nsecureface
{
    class FaceTool
    {
        
    public:
        
        FaceTool(): label_count(1)
        {
            
        }
        
        void LoadJsonConfig(std::string config_file_path);
        void StartRecognizeService();
        void Debug();
        void LaunchTestClient();
        void Close();
    private:
        void LoadCaffeDetector();
        void CreateEmbeddings();
        void LoadEmbedder();
        std::string GetLabelName(int label);
        
        NSecureFaceConfig config;
        cv::dnn::Net detector;
        cv::dnn::Net embedder;
        
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        
        //cv::face::LBPHFaceRecognizer* recognizer;
        cv::Ptr<cv::face::FaceRecognizer> recognizer;
        std::unordered_map<std::string, int> label_map;
        int label_count;
    };
}
#endif /* face_processing_hpp */
