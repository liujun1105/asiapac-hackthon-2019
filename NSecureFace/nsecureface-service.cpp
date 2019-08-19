#include "face-processing.hpp"

using namespace nsecureface;

int main(int argc, char** argv)
{
    FaceTool face_tool;
    face_tool.LoadJsonConfig(argv[1]);
    face_tool.Debug();
    face_tool.StartRecognizeService();
    face_tool.LaunchTestClient();
//    face_tool.AddTrainingData("/Users/junliu/Desktop/liuyunuo", "liuyunuo");
//    face_tool.AddTrainingData("/Users/junliu/Desktop/yanglifang", "yanglifang");

//    face_tool.RecognizeFromImages("/Users/junliu/Desktop/liuyucheng");
//    face_tool.RecognizeFromImages("/Users/junliu/Desktop/yanglifang");
//    face_tool.RecognizeFromImages("/Users/junliu/Desktop/liuyunuo");
//    face_tool.RecognizeFromImages("/Users/junliu/Desktop/liujunju");
//    face_tool.RecognizeFromImages("/Users/junliu/Desktop/test");
    
    return 0;
}
