//
//  face-processing.cpp
//  NSecureFaceService
//
//  Created by Jun Liu on 2019/8/17.
//

#include "face-processing.hpp"

#include <fstream>
#include <string>
#include <stdio.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/face.hpp>
#include "json.hpp"
#include "cppfs/fs.h"
#include "cppfs/FileHandle.h"
#include "cppfs/FilePath.h"
#include <opencv2/face/facemarkLBF.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/objdetect.hpp"
#include <sstream>

using json = nlohmann::json;

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace cv::ml;
using namespace cppfs;
using namespace std;

namespace nsecureface
{
    void FaceTool::Debug()
    {
        printf("%s\n", std::string(110, '#').c_str());
        printf("%-30s: %d\n", "Device", config.device);
        printf("%-30s: %s\n", "Face Images", config.face_images.c_str());
        printf("%-30s: %s\n", "Embeddings", config.face_embeddings.c_str());
        printf("%-30s: %s\n", "Recognizer", config.face_recognizer.c_str());
        printf("%-30s: %s\n", "Labels", config.face_labels.c_str());
        printf("%-30s: %s\n", "DNN Network", config.dnn_network.c_str());
        printf("%-30s: %s\n", "DNN Weights", config.dnn_weights.c_str());
        printf("%-30s: %s\n", "Detector Directory", config.face_model.c_str());
        printf("%s\n", std::string(110, '#').c_str());
    }
    
    void FaceTool::LoadJsonConfig(std::string config_file_path)
    {
        std::ifstream config_reader(config_file_path);
        json json_config;
        
        if (config_reader.is_open())
        {
            config_reader >> json_config;
            config_reader.close();
            
            
            config.device = json_config["device"];
            config.face_images = json_config["face_images"];
            config.face_embeddings = json_config["face_embeddings"];
            config.face_recognizer = json_config["face_recognizer"];
            config.face_labels = json_config["face_labels"];
            config.face_model = json_config["face_model"];
            config.dnn_network = json_config["face_detector"]["network"];
            config.dnn_weights = json_config["face_detector"]["weights"];
        }
        else {
            printf("[ERROR] failed to open configuration file config.json");
        }
    }
    
    void FaceTool::StartRecognizeService()
    {
        LoadCaffeDetector();
        LoadEmbedder();
        CreateEmbeddings();
    }
    
    void FaceTool::LoadCaffeDetector()
    {
        cout << "start loading caffe detector" << endl;
        printf("reading network file from %s\n", config.dnn_network.c_str());
        printf("reading weights file from %s\n", config.dnn_weights.c_str());
        detector = readNetFromCaffe(config.dnn_network, config.dnn_weights);
        cout << "complete loading caffe detector" << endl;
    }
    
    void FaceTool::LoadEmbedder()
    {
        cout << "loading embedder" << endl;
        embedder = readNetFromTorch(config.face_model);
        cout << "complete loading embedder" << endl;
    }
    
    void FaceTool::CreateEmbeddings()
    {
        cout << "start creating embeddings" << endl;
        FileHandle dir = fs::open(config.face_images);
        
        if (dir.exists() && dir.isDirectory())
        {
            dir.traverse([this](FileHandle& fh) -> bool {
                FilePath fp(fh.path());
                if (fp.extension() == ".jpeg" || fp.extension() == ".png" || fp.extension() == ".jpg")
                {
                    string last_dirname = fp.directoryPath();
                    last_dirname.replace(0, config.face_images.size() + 1, "");
                    if (last_dirname[last_dirname.size()-1] == '/')
                        last_dirname.replace(last_dirname.size()-1, 1, "");
                    
                    auto is = fh.createInputStream();
                    Mat img = imread(fp.path());

                    Mat blobImg = blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
                    detector.setInput(blobImg);
                    Mat detection = detector.forward();
                    
                    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
                    for(int i = 0; i < detectionMat.rows; i++)
                    {
                        float confidence = detectionMat.at<float>(i, 2);
                        
                        if(confidence > 0.7)
                        {
                            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
                            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
                            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
                            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

                            Rect face_rect(cv::Point(x1, y1), cv::Point(x2, y2));
                            if (face_rect.height < 20 || face_rect.width < 20)
                                continue;
                            
                            Mat ROI = img.clone();
                            if (face_rect.x >= 0 && face_rect.y >= 0 && face_rect.y + face_rect.height < ROI.rows && face_rect.x + face_rect.width < ROI.cols)
                            {
                                ROI(face_rect);
                            }
                            assert(ROI.rows >= 0);
                            assert(ROI.cols >= 0);
//                            Mat face = blobFromImage(ROI, 1.0 / 255, Size(96, 96), Scalar(0, 0, 0), true, false);
//                            cout << face.size << endl;
//
//                            embedder.setInput(face);
//                            Mat embeddings = embedder.forward().clone();
//                            cout << embeddings.size << endl;
                            cvtColor(ROI, ROI, COLOR_RGB2GRAY);
                            images.push_back(ROI);
                            if (this->label_map[last_dirname] == 0)
                            {
                                this->label_map[last_dirname] = label_count++;
                            }
                            labels.push_back(this->label_map[last_dirname]);
                            
                            printf(
                               "label = %d, name = %s, color = %d, size=%dx%d\n",
                               this->label_map[last_dirname], last_dirname.c_str(), ROI.channels(), ROI.rows, ROI.cols
                            );
                        }
                    }
                }
                return true; // continue
            });
            
            printf("#embeddings = %d, #labels = %d\n", static_cast<int>(images.size()), static_cast<int>(labels.size()));
            
            this->recognizer = LBPHFaceRecognizer::create(1, 5, 5, 5, 40);
            this->recognizer->train(images, labels);
        }
        cout << "complete creating embeddings" << endl;
    }
    
    void FaceTool::LaunchTestClient()
    {
        VideoCapture capture(config.device);
        if (capture.isOpened())
        {
            Mat frame;
            namedWindow("Face Recognition", WINDOW_NORMAL);
            while (true)
            {
                capture >> frame;
                
                if (frame.empty())
                {
                    break;
                }
            
                Mat blobImg = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
                
                detector.setInput(blobImg);
                Mat detection = detector.forward();
                
                Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
                for(int i = 0; i < detectionMat.rows; i++)
                {
                    float confidence = detectionMat.at<float>(i, 2);
                    
                    if(confidence > 0.7)
                    {
                        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                        Rect face_rect(cv::Point(x1, y1), cv::Point(x2, y2));
                        Mat ROI(frame, face_rect);
                        
//                        Mat face = blobFromImage(ROI, 1.0 / 255, Size(96, 96), Scalar(0, 0, 0), true, false);
//
//                        embedder.setInput(face);
//                        Mat embeddings = embedder.forward();
 
                        int label = -1;
                        double confidence = 0.0;
                        cvtColor(ROI, ROI, COLOR_RGB2GRAY);
                        this->recognizer->predict(ROI, label, confidence);
                        
                        stringstream ss;
                        
                        if (label > 0 && confidence < this->recognizer->getThreshold()) {
                            printf("person index = %d, label = %d, confidence level = %f\n", i, label, confidence);
                            int y = y1 - 10 > 10 ?  y1 - 10 : y1 + 10;
                            rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255));
                            ss << "Person#" << i << " " << GetLabelName(label) << " " << confidence;
                            putText(frame, ss.str(), cv::Point(x1, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 2);
                        } else {
                            int y = y1 - 10 > 10 ?  y1 - 10 : y1 + 10;
                            rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 255));
                            ss << "Person#" << i << " " << "Unknown";
                            putText(frame, ss.str(), cv::Point(x1, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 2);
                        }
                    }
                }
                
                imshow("Face Recognition", frame);
                
                int k = waitKey(27);
                if (k == 27) break;
            }
            frame.release();
            capture.release();
        }
        else
        {
            fprintf(stderr, "failed to launch device #%d\n", config.device);
        }
    }
    
    void FaceTool::Close()
    {
        this->recognizer->clear();
        for (auto img : images)
            img.release();
    }
    
    string FaceTool::GetLabelName(int label)
    {
        for (auto iter = this->label_map.begin(); iter != this->label_map.end(); iter++)
        {
            if (iter->second == label)
            {
                return iter->first;
            }
        }
        
        return nullptr;
    }
}
