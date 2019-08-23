//
//  face-processing.hpp
//  NSecureFaceService
//
//  Created by Jun Liu on 2019/8/17.
//

#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <thread>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/face/facemarkLBF.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include "cppfs/fs.h"
#include "cppfs/FileHandle.h"
#include "cppfs/FilePath.h"

#ifdef _WIN32
#include <Windows.h>
#include <WinUser.h>
#include <Lmcons.h>
#endif

#include "client.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace cv::ml;
using namespace std;
using namespace std::chrono;
using namespace cppfs;

namespace nsecureface
{
#ifdef _WIN32
    LRESULT CALLBACK WindowProc(HWND hwnd, UNIT uMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (uMsg)
        {
            case WM_DESTROY:
                PostQuitMessage(0);
                return 0;
            case WM_PAINT: 
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);

                auto dpiScale = GetDeviceCaps(hdc, LOGPIXELSX) / 60.0;
                auto fontSize = 80;

                RECT r;
                r.top = 0;
                r.bottom = int(round(dpiScale * fontSize));
                r.left = 0;
                r.right = 150;

                auto rect = GetClientRect(hwnd, &r);

                FillRect(hdc, &ps.rcPaint, (HBRUSH)(COLOR_WINDO + 1));

                // DrawText(hdc, "haha", rect, NULL, DT_CENTER | DT_NOCLIP | DT_SINGLELINE | DT_VCENTER);

                EndPaint(hwnd, &ps);
            }
            return 0;
            case WM_CLOST:
                DestroyWindow(hwnd);
            default:
                return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }

        return TRUE;
    }
#endif
        
    NSecureFaceClient::NSecureFaceClient(NSecureFaceConfig config)
    {
        this->config = config;
        this->label_count = 1;
        this->pause = true;
    }
        
    string NSecureFaceClient::GetLabelName(int label)
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
    
    void NSecureFaceClient::TrainRecognizer()
    {
        
        cout << "start training recognizer" << endl;
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
                                ROI.release();
                                ROI = Mat(img, face_rect);
                            }
                            assert(ROI.rows >= 0);
                            assert(ROI.cols >= 0);
                            
                            cvtColor(ROI, ROI, COLOR_RGB2GRAY);
                            images.push_back(ROI);
                            if (this->label_map[last_dirname] == 0)
                            {
                                this->label_map[last_dirname] = label_count++;
                            }
                            labels.push_back(this->label_map[last_dirname]);
                            
                            Mat rotated = ROI.clone();
                            transpose(rotated, rotated);
                            flip(rotated, rotated,1);
                            images.push_back(rotated);
                            labels.push_back(this->label_map[last_dirname]);
                            
                            rotated = ROI.clone();
                            transpose(rotated, rotated);
                            flip(rotated, rotated,0);
                            images.push_back(rotated);
                            labels.push_back(this->label_map[last_dirname]);
                            
                            rotated = ROI.clone();
                            flip(rotated, rotated,-1);
                            images.push_back(rotated);
                            labels.push_back(this->label_map[last_dirname]);
                            
                            
                            for (int i=1; i<4; i++)
                            {
                                Mat rotated = ROI.clone();
                                Point2f src_center(rotated.cols/2.0F, rotated.rows/2.0F);
                                Mat dest;
                                cv::warpAffine(rotated, dest, cv::getRotationMatrix2D(src_center, (i%4)*90, 1.0), rotated.size());
                                images.push_back(dest);
                                labels.push_back(this->label_map[last_dirname]);
                            }
                            
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
            
            this->recognizer->train(images, labels);
        }
        cout << "complete training recognizer" << endl;
        
    }
    
    void NSecureFaceClient::PerformFaceAlignment(int& label, double& distance, Mat& frame, Rect& face_rect)
    {
        vector< vector<Point2f> > landmarks;
        vector<Rect> faces = { face_rect };
        bool success = this->facemark_detector->fit(frame, faces, landmarks);
        if (success)
        {
            for (auto landmark : landmarks)
            {
                Mat output = frame.clone();
                
                for (auto landmark_point : landmark)
                {
                    cv::circle(output, landmark_point, 1, Scalar(255, 0, 0), FILLED);
                }
                imshow("Landmarks", output);
                
                Point2f left_eye((landmark[39].x - landmark[36].x) / 2.0 + landmark[36].x, (landmark[39].y - landmark[36].y) / 2.0 + landmark[36].y);
                Point2f right_eye((landmark[45].x - landmark[42].x) / 2.0 + landmark[42].x, (landmark[45].y - landmark[42].y) / 2.0 + landmark[42].y);
                
                Point2f eye_center((left_eye.x + right_eye.x)/2.0F, (left_eye.y + right_eye.y)/2.0F);
                double dy = (right_eye.y - left_eye.y);
                double dx = (right_eye.x - left_eye.x);
                double len = sqrt(dx * dx + dy * dy);
                
                float angle = atan2(dy, dx) * 180.0/CV_PI;
                
                const double DESIRED_LEFT_EYE_X = 0.16;
                const double DESIRED_RIGHT_EYE_X = (1.0F - DESIRED_LEFT_EYE_X);
                
                const int DESIRED_FACE_WIDTH = face_rect.width;
                const int DESIRED_FACE_HEIGHT = face_rect.height;
                
                double desired_length = (DESIRED_RIGHT_EYE_X - 0.16);
                double scale = desired_length * DESIRED_FACE_WIDTH / len;
                
                Mat r = getRotationMatrix2D(eye_center, angle, scale);
                double ex = DESIRED_FACE_WIDTH * 0.5f - eye_center.x;
                double ey = DESIRED_FACE_HEIGHT * 0.14 - eye_center.y;
                
                r.at<double>(0, 2) += ex;
                r.at<double>(1, 2) += ey;
                
                Mat warped = Mat(DESIRED_FACE_HEIGHT, DESIRED_FACE_WIDTH, CV_8U, Scalar(128));
                
                warpAffine(output, warped, r, warped.size());
                
                cvtColor(warped, warped, COLOR_RGB2GRAY);
                this->recognizer->predict(warped, label, distance);
                
                imshow("Alignment", warped);
            }
        }
    }
    
    void NSecureFaceClient::LaunchTestClient()
    {
        
        this->detector = readNetFromCaffe(config.dnn_network, config.dnn_weights);
        this->facemark_detector = FacemarkLBF::create();
        this->facemark_detector->loadModel(config.facemark_detector);
        
        this->embedder = readNetFromTorch(config.face_model);
        
        this->recognizer = LBPHFaceRecognizer::create(1, 5, 5, 5, 40);
        this->TrainRecognizer();
        
        VideoCapture capture(config.device);
        if (capture.isOpened())
        {
            stringstream ss;
            Mat frame;
            
            namedWindow("NSecureFace Client", WINDOW_NORMAL);
            resizeWindow("NSecureFace Client", 640, 480);
            
            namedWindow("Alignment", WINDOW_NORMAL);
            resizeWindow("Alignment", 160, 240);
            
            namedWindow("Landmarks", WINDOW_NORMAL);
            resizeWindow("Landmarks", 640, 480);

            namedWindow("Grayed", WINDOW_NORMAL);
            resizeWindow("Grayed", 160, 240);
            
            namedWindow("Transpose + Flip 1", WINDOW_NORMAL);
            resizeWindow("Transpose + Flip 1", 160, 240);
            
            namedWindow("Transpose + Flip 2", WINDOW_NORMAL);
            resizeWindow("Transpose + Flip 2", 160, 240);

            namedWindow("Transpose + Flip 3", WINDOW_NORMAL);
            resizeWindow("Transpose + Flip 3", 160, 240);
            
            namedWindow("Affine Transformation 1", WINDOW_NORMAL);
            resizeWindow("Affine Transformation 1", 160, 240);
            
            namedWindow("Affine Transformation 2", WINDOW_NORMAL);
            resizeWindow("Affine Transformation 2", 160, 240);

            namedWindow("Affine Transformation 3", WINDOW_NORMAL);
            resizeWindow("Affine Transformation 3", 160, 240);
            
            namedWindow("Affine Transformation 4", WINDOW_NORMAL);
            resizeWindow("Affine Transformation 4", 160, 240);

            
            while (true)
            {
                capture >> frame;
                
                if (frame.empty())
                {
                    break;
                }
                
                Mat original_frame = frame.clone();
                
                ss.str("");
                ss.clear();
                ss << "Paused = " << pause;
                putText(frame, ss.str(), cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 2);
                
                Mat blobImg = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
                
                detector.setInput(blobImg);
                Mat detection = detector.forward();
                
                Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
                bool has_permission = false;
                if (detectionMat.rows == 0) has_permission = 0;

                for(int i = 0; i < detectionMat.rows; i++)
                {
                    float confidence = detectionMat.at<float>(i, 2);
                    
                    if(confidence > 0.8)
                    {
                        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                        
                        Rect face_rect(cv::Point(x1, y1), cv::Point(x2, y2));

                        Mat ROIDemo = frame.clone();
                        if (face_rect.x >= 0 && face_rect.y >= 0 && face_rect.y + face_rect.height < ROIDemo.rows && face_rect.x + face_rect.width < ROIDemo.cols)
                        {
                            ROIDemo.release();
                            ROIDemo = Mat(frame, face_rect);
                        }
                        
                        cvtColor(ROIDemo, ROIDemo, COLOR_RGB2GRAY);
                        imshow("Grayed", ROIDemo);
                        
                        Mat rotated = ROIDemo.clone();
                        transpose(rotated, rotated);
                        flip(rotated, rotated,1);
                        imshow("Transpose + Flip 1", rotated);
                        
                        rotated = ROIDemo.clone();
                        transpose(rotated, rotated);
                        flip(rotated, rotated,0);
                        imshow("Transpose + Flip 2", rotated);
                        
                        rotated = ROIDemo.clone();
                        flip(rotated, rotated,-1);
                        imshow("Transpose + Flip 3", rotated);
                        
                        
                        for (int i=1; i<4; i++)
                        {
                            Mat rotated = ROIDemo.clone();
                            Point2f src_center(rotated.cols/2.0F, rotated.rows/2.0F);
                            Mat dest;
                            cv::warpAffine(rotated, dest, cv::getRotationMatrix2D(src_center, (i%4)*90, 1.0), rotated.size());
                            imshow("Affine Transformation " + std::to_string(i), dest);
                        }
                            
                        
                        int label = -1;
                        double confidence = 0.0;
                        PerformFaceAlignment(label, confidence, frame, face_rect);
                        printf("[Face Alignment] person index = %d, label = %d, confidence level = %f\n", i, label, confidence);
                        
                        Mat ROI = frame.clone();
                        if (face_rect.x >= 0 && face_rect.y >= 0 && face_rect.y + face_rect.height < ROI.rows && face_rect.x + face_rect.width < ROI.cols)
                        {
                            ROI.release();
                            ROI = Mat(frame, face_rect);
                        }
                        
                        
                        cvtColor(ROI, ROI, COLOR_RGB2GRAY);
                        this->recognizer->predict(ROI, label, confidence);
                        
                        
                        ss.str("");
                        ss.clear();
                        if (label > 0 && confidence < 20) {
                            printf("person index = %d, label = %d, confidence level = %f\n", i, label, confidence);
                            int y = y1 - 10 > 10 ?  y1 - 10 : y1 + 10;
                            rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255));
                            ss << "Person#" << i << " " << GetLabelName(label) << " " << confidence;
                            putText(frame, ss.str(), cv::Point(x1, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 2);

#ifdef _WIN32
                            char username[UNLEN + 1];
                            DWORD username_len = UNLEN + 1;
                            GetUserName(username, &username_len);

                            if (GetLabelName(label) == username)
                            {
                                has_permission = true;
                            }

#endif                            

                        } else {
                            int y = y1 - 10 > 10 ?  y1 - 10 : y1 + 10;
                            rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), Scalar(0, 0, 255));
                            ss << "Person#" << i << " " << "Unknown";
                            putText(frame, ss.str(), cv::Point(x1, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 2);
                            
                            if (!pause)
                            {
                                milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
                                ss << config.face_capture_unknown << "unknown_" << ms.count() << ".jpg";
                                imwrite(ss.str(), original_frame);
                            }
                        }
                    }
                }

                if (has_permission) {
                    thread t([&](){GrantAccess();});
                    t.detach();
                } else if (!has_permission) {
                    thread t([&](){BlockAccess();});
                    t.detach();
                }
                
                imshow("NSecureFace Client", frame);
                
                int k = waitKey(27);
                if (k == 27) break;
                else if (k == 'c' && !pause)
                {
                    ss.str("");
                    ss.clear();
                    milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
                    ss << config.face_capture_negative << "negative_" << ms.count() << ".jpg";
                    imwrite(ss.str(), original_frame);
                }
                else if (k == 'p')
                {
                    pause = !pause;
                }
            }
            frame.release();
            capture.release();
        }
        else
        {
            fprintf(stderr, "failed to launch device #%d\n", config.device);
        }
        
        destroyAllWindows();
    }

    void NSecureFaceClient::BlockAccess()
    {
#ifdef _WIN32
        const char* CLASS_NAME = "NSecureFace";

        WNDCLASS wc = {};
        wc.hInstance = GetModuleHandle("NSecureFace");
        wc.lpszClassName = CLASS_NAME;
        wclpfnWndProc = nsecureface::WindowProc;

        RegisterClass(&wc);

        int win_style_opt = WS_EX_COMPOSITED | WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TRANSPARENT;
        int win_style = WS_DISABLED | WS_POPUP | WS_VISIBLE;

        HWND hwnd = CreateWindowEx(
            win_style_opt, // Optional window styles
            CLASS_NAME,    // Window class
            "Some Text Here", // Window text
            win_style,

            // Size and position
            CW_USEDEFAULT, CW_USEDFAULT, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSREEN),

            NULL,           // Parent
            NULL,           // Menu
            wc.hInstance,   // Instance handle
            NULL            // Additional application data
        );

        SetLayeredWindowAttributes(hwnd, 0x00ffefaf, 245, LWA_COLORKEY | LWA_ALPHA);

        UpdateWindow(hwnd);

        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);

        ShowWindow(hmwd, SW_SHOW);

        MSG msg = { };
        while (GetMessage(&msg, NULL, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
#endif
    }

    void NSecureFaceClient::GrantAccess()
    {
#ifdef _WIN32
        HWND hwnd = FindWindow("NSecureFace", 0);

        if (hwnd != NULL)
        {
            cout << "find window" << endl;
            SendMessage(hwnd, WM_CLOSE, SC_CLOSE, 0);
            hwnd = NULL;
        }
        else
        {
            cout << "cannot find window" << endl;
        }
#endif    
    }
}

