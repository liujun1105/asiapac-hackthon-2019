#include "face-processing.hpp"
#include "client.hpp"

#include <thread>

#include <nng.h>
#include <transport/tcp/tcp.h>
#include <protocol/reqrep0/rep.h>
#include <protocol/reqrep0/req.h>

#include "common.hpp"

#include "httplib.h"

using namespace nsecureface;
using namespace std;

void LaunchImageServer(const char *url, httplib::Client client)
{
    nng_socket sock;
    int        rv;
    
    if ((rv = nng_rep0_open(&sock)) != 0)
    {
        fatal("nng_rep0_open", rv);
    }
    else
    {
        printf("socket opened\n");
    }
    
    if ((rv = nng_listen(sock, url, NULL, 0)) != 0)
    {
        fatal("nng_listen", rv);
    }
    else
    {
        printf("listening to %s\n", url);
    }
    
    for (;;) {
        char *   buf = NULL;
        size_t   sz;
        
        if ((rv = nng_recv(sock, &buf, &sz, NNG_FLAG_ALLOC)) != 0) {
            fatal("nng_recv", rv);
        }
        
        if (sz > 0)
        {
            printf("[server] # bytes received => %d\n", static_cast<int>(sz));

            httplib::Params params;
            params.emplace("image", string(buf));
            params.emplace("length", std::to_string(sz));
            auto res = client.Post("/recognize", params);
            cout << res->status << endl;
            
            string identity = res->body;
            cout << identity << endl;
            
            if ((rv = nng_send(sock, (void *)identity.c_str(), identity.size()+1, NNG_FLAG_NONBLOCK)) != 0)
            {
                fatal("nng_send", rv);
            }
        }
        
        nng_free(buf, sz);
    }
}

void LaunchImageServer(NSecureFaceConfig config)
{
    httplib::Client client(config.face_recognition_service.c_str(), config.face_recognition_service_port);
    LaunchImageServer(config.image_server_url.c_str(), client);
    cout << "face image server service is up" << endl;
}

void LaunchFaceRecognitionService(NSecureFaceConfig config)
{
    std::thread httpsvr_thread([&config=config]{
        FaceTool face_tool(config);
        
        cout << "start recognition service" << endl;
        face_tool.StartRecognizeService();
        cout << "recognition service is up" << endl;
        
        using namespace httplib;
        
        Server httpsvr;
        
        httpsvr.Get("/status", [](const Request& req, Response& res) {
            res.set_content("Started", "text/plain");
        });
        
        httpsvr.Post("/recognize", [&](const Request& req, Response& res) {
            cv::Mat img_decode;
            string identity = "unknown";
            
            if (req.has_param("image") && req.has_param("length"))
            {
            
                cout << req.get_param_value("image") << endl;
                cout << req.get_param_value("length") << endl;
                
                string buf = req.get_param_value("image");
                int sz = stoi(req.get_param_value("length"));
//                string img_str(buf, sz);
                printf("[server] encoded string size => %d\n", buf.length());
                
                std::vector<uchar> data(buf.begin(), buf.end());
                printf("[server] vector size => %d\n", static_cast<int>(data.size()));
                img_decode = cv::imdecode(data, 1);
                
                identity = face_tool.RecognizeFromImage(img_decode);
            }
            res.set_content(identity.c_str(), identity.length(), "text/plain");
        });
        
        printf("face recognition server is up, listening to %s:%d\n", config.face_recognition_service.c_str(), config.face_recognition_service_port);
        httpsvr.listen(config.face_recognition_service.c_str(), config.face_recognition_service_port);
    });
    
//    std::thread t([&config=config]{
        LaunchImageServer(config);
//    });
    
//    httpsvr_thread.join();
//    t.join();
}

void LaunchClient(const char *url)
{
    nng_socket sock;
    int        rv;
    size_t     sz;
    char *     buf = NULL;
    
    if ((rv = nng_req0_open(&sock)) != 0)
    {
        fatal("nng_socket", rv);
    }
    
    if ((rv = nng_dial(sock, url, NULL, 0)) != 0)
    {
        fatal("nng_dial", rv);
    }
}

int main(int argc, char** argv)
{
    string app_type(argv[1]);
    NSecureFaceConfig config = nsecureface::LoadJsonConfig(argv[2]);
    nsecureface::Debug(config);
    
    if (app_type == "service")
    {
        LaunchFaceRecognitionService(config);
    }
    else if (app_type == "client")
    {
        NSecureFaceClient client(config);
        client.LaunchTestClient();
    }
    else if (app_type == "test-client")
    {
        FaceTool face_tool(config);
        face_tool.StartRecognizeService();
        face_tool.LaunchTestClient();
    }
    else if (app_type == "window")
    {
        NSecureFaceClient client(config);
        client.BlockAccess();
    }
    
    return 0;
}
