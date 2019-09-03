#ifndef NSECUREFACE_NETWORK_H
#define NSECUREFACE_NETWORK_H

#include <nng/nng.h>
#include <nng/supplemental/http/http.h>
#include <stdio.h>
#include <stdlib.h>

namespace nsecureface
{
    namespace network
    {
        class ImageSender
        {
        public:
            bool Init(const char* url_str)
            {

                nng_url* url;

                int rv;

                if ((rv == nng_url_parse(&url, url_str)) != 0) 
                {
                    fatal("nng_url_parse", rv)
                    return false;
                }

                if ((rv == nng_http_client_alloc(&client, url)) != 0) 
                {
                    fatal("nng_http_client_alloc", rv)
                    return false;
                }
                
                return true;
            }

            bool SendImage()
            {
                nng_http_conn* conn;
                nng_aio* aio;
                nng_http_req* req;
                nng_http_res* res;

                int rv;

                if ((rv == nng_aio_alloc(&aio, NULL, NULL)) != 0) 
                {
                    fatal("nng_aio_alloc", rv)
                    return false;
                }

                if ((rv == nng_http_req_alloc(&req, url)) != 0) 
                {
                    fatal("nng_http_req_alloc", rv)
                    return false;
                }

                if ((rv == nng_http_res_alloc(&res)) != 0) 
                {
                    fatal("nng_http_res_alloc", rv)
                    return false;
                }
            }
            
        private:
            nng_http_client *client;
            
            void fatal(const char *func, int rv)
            {
                fprintf(stderr, "%s: %s\n", func, nng_strerror(rv));
                exit(1);
            }
        };
    }
}

#endif /* NSECUREFACE_NETWORK_H */
