//
//  socket.h
//  NSecureFace
//
//  Created by Jun Liu on 2019/9/3.
//

#ifndef NSECUREFACE_SOCKET_H
#define NSECUREFACE_SOCKET_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <nng.h>
#include <protocol/reqrep0/req.h>

namespace nsecureface
{
    namespace socket
    {
        class StreamingClient
        {
        public:
            bool Init(const char *url)
            {
                int rv;
                if ((rv = nng_req0_open(&sock)) != 0)
                {
                    fatal("nng_req0_open", rv);
                    return false;
                }
                
                if ((rv = nng_dial(sock, url, NULL, 0)) < 0)
                {
                    fatal("nng_dial", rv);
                    return false;
                }
                
                return true;
            }
            
            void Close()
            {
                nng_close(sock);
            }
            
            void Send(void* data, size_t size)
            {
                void* buf = nng_alloc(size);
                memcpy(buf, data, size);
                
                int rv;
                
                if ((rv = nng_send(sock, buf, size, NNG_FLAG_ALLOC)) != 0)
                {
                    fatal("nng_send", rv);
                }
                
                nng_free(buf, size);
            }
            
        private:
            nng_socket sock;
            
            void fatal(const char *func, int rv)
            {
                fprintf(stderr, "%s: %s\n", func, nng_strerror(rv));
                exit(1);
            }
        };
    }
}

#endif /* NSECUREFACE_SOCKET_H */
