#ifndef NSECUREFACE_NETWORK_H
#define NSECUREFACE_NETWORK_H

#include <nng.h>
#include <supplemental/http/http.h>
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

                if ((rv = nng_url_parse(&url, url_str)) != 0) 
                {
					fatal("nng_url_parse", rv);
                    return false;
                }

                if ((rv = nng_http_client_alloc(&client, url)) != 0) 
                {
					fatal("nng_http_client_alloc", rv);
                    return false;
                }
                
                return true;
            }

			bool SendImage(const char* target_url, void* data, size_t size)
			{
				nng_http_conn* conn;
				nng_aio* aio;
				nng_http_req* req;
				nng_http_res* res;

				int rv;

				if ((rv = nng_aio_alloc(&aio, NULL, NULL)) != 0)
				{
					fatal("nng_aio_alloc", rv);
					return false;
				}

				if ((rv = nng_http_req_alloc(&req, NULL)) != 0)
				{
					fatal("nng_http_req_alloc", rv);
					return false;
				}

				if ((rv = nng_http_res_alloc(&res)) != 0)
				{
					fatal("nng_http_res_alloc", rv);
					return false;
				}

				// start connection
				nng_http_client_connect(client, aio);
				nng_aio_wait(aio);
				if ((rv = nng_aio_result(aio)) != 0)
				{
					fatal("nng_aio_result", rv);
					return false;
				}
				conn = (nng_http_conn*)nng_aio_get_output(aio, 0);

				if ((rv = nng_http_req_set_data(req, data, size)) != 0)
				{
					fatal("nng_http_req_set_data", rv);
					return false;
				}
				nng_http_req_set_uri(req, target_url);
				nng_http_conn_write_req(conn, req, aio);
				nng_aio_wait(aio);

				if ((rv = nng_aio_result(aio)) != 0)
				{
					fatal("nng_aio_result", rv);
					return false;
				}

				nng_http_conn_read_res(conn, res, aio);
				nng_aio_wait(aio);

				if ((rv = nng_aio_result(aio)) != 0)
				{
					fatal("nng_aio_result", rv);
					return false;
				}

				if (nng_http_res_get_status(res) != NNG_HTTP_STATUS_OK)
				{
					fprintf(stderr, "HTTP Server Responded: %d %s\n", nng_http_res_get_status(res), nng_http_res_get_reason(res));
					return false;
				}

				const char* hdr;
				if ((hdr = nng_http_res_get_header(res, "Content-Length")) == NULL)
				{
					fprintf(stderr, "Missing Content-Length Header.\n");
					return false;
				}

				int len = atoi(hdr);
				if (len == 0)
				{
					return false;
				}

				void* response_data = malloc(len);

				// setup a single iov to point to the buffer
				nng_iov iov;
				iov.iov_len = len;
				iov.iov_buf = response_data;

				// Following never fails with fewer than 5 elements
				nng_aio_set_iov(aio, 1, &iov);

				nng_http_conn_read_all(conn, aio);
				nng_aio_wait(aio);
				if ((rv = nng_aio_result(aio)) != 0)
				{
					fatal("nng_aio_result", rv);
					return false;
				}

				fwrite(data, 1, len, stdout);
				nng_http_req_free(req);
				nng_http_res_free(res);
				nng_http_conn_close(conn);
				nng_aio_free(aio);

				return true;
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
