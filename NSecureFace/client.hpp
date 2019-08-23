//
//  face-processing.hpp
//  NSecureFaceService
//
//  Created by Jun Liu on 2019/8/17.
//

#ifndef NSECUREFACE_CLIENT_H
#define NSECUREFACE_CLIENT_H

#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>


#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include "nsecureface.h"


namespace nsecureface
{
    class NSecureFaceClient
    {
    private:
        int label_count;
        bool pause;
        
        NSecureFaceConfig config;
        
        cv::dnn::Net detector;
        cv::dnn::Net embedder;
        
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        
        cv::Ptr<cv::face::Facemark> facemark_detector;
        cv::Ptr<cv::face::FaceRecognizer> recognizer;
        
        std::unordered_map<std::string, int> label_map;
        
    public:
        
        NSecureFaceClient(NSecureFaceConfig config);
        
        std::string GetLabelName(int label);
        
        void TrainRecognizer();

        
        void PerformFaceAlignment(int& label, double& distance, cv::Mat& frame, cv::Rect& face_rect);
        
        
        void LaunchTestClient();
        void BlockAccess();
        void GrantAccess();
    };
}
#endif /* NSECUREFACE_CLIENT_H */
