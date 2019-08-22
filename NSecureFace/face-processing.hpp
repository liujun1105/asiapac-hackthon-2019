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
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include "nsecureface.h"

namespace nsecureface
{
    class FaceTool
    {
        
    public:
        
        FaceTool(NSecureFaceConfig& config);
        
        void InitRecognizer();
        void InitCaffeDetector();
        void InitEmbedder();
        
        void StartRecognizeService();
        void LaunchTestClient();
        void Close();
        void AddTrainingData(std::string dir_path, std::string label_name);
        
        bool IsRecognizerInitialized();
        bool IsEmbedderInitialized();
        bool IsDetectorInitialized();
        
        void RecognizeFromImages(std::string dir_path);
        std::string RecognizeFromImage(cv::Mat image);
    private:
        void CreateDirectories();
        void PerformFaceAlignment(int& label, double& distance, cv::Mat& frame, cv::Rect& face_rect);
        void TrainRecognizer();
        
        std::string GetLabelName(int label);
        
        NSecureFaceConfig config;
        
        cv::dnn::Net detector;
        cv::dnn::Net embedder;
        
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        
        cv::Ptr<cv::face::Facemark> facemark_detector;
        cv::Ptr<cv::face::FaceRecognizer> recognizer;
        
        std::unordered_map<std::string, int> label_map;
        
        int label_count;
        
        bool recognizer_initialized;
        bool embedder_initialized;
        bool detector_initialized;
        bool pause;
    };
}
#endif /* face_processing_hpp */
