#include "face-processing.hpp"

using namespace nsecureface;
#undef CV_TRACE_FUNCTION
int main(int argc, char** argv)
{
    FaceTool face_tool;
    face_tool.LoadJsonConfig(argv[1]);
    face_tool.Debug();
    face_tool.StartRecognizeService();
    face_tool.LaunchTestClient();
    
//    std::vector<cv::Mat> images;
//    std::vector<int> labels;
//    
//    images.push_back(cv::imread("/Users/junliu/Development/Project/asiapac-hackthon-2019/NSecureFace/data/face-images/yanglifang/yanglifang_2.png", cv::IMREAD_GRAYSCALE));
//    labels.push_back(1);
//    images.push_back(cv::imread("/Users/junliu/Development/Project/asiapac-hackthon-2019/NSecureFace/data/face-images/liujunju/liujunju_59.png", cv::IMREAD_GRAYSCALE));
//    labels.push_back(2);
//    images.push_back(cv::imread("/Users/junliu/Development/Project/asiapac-hackthon-2019/NSecureFace/data/face-images/liujunju/照片 490.jpg", cv::IMREAD_GRAYSCALE));
//    labels.push_back(2);
//    for (auto image : images) {
//        std::cout << image.channels() << std::endl;
//        std::cout << image.size << std::endl;
//        std::cout << image.elemSize() << std::endl;
//    }
//    
//    cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
//    recognizer->train(images, labels);
    
    return 0;
}
