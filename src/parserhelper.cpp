
#include "http_parser.hpp"
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <fstream>
#include <csignal>
#include <ctime>
#include <regex>
#include <cctype>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <condition_variable>

size_t
strlncat(char *dst, size_t len, const char *src, size_t n)
{
    size_t slen;
    size_t dlen;
    size_t rlen;
    size_t ncpy;

    slen = strnlen(src, n);
    dlen = strnlen(dst, len);

    if (dlen < len) {
        rlen = len - dlen;
        ncpy = slen < rlen ? slen : (rlen - 1);
        memcpy(dst + dlen, src, ncpy);
        dst[dlen + ncpy] = '\0';
    }

    //assert(len > slen + dlen);
    return slen + dlen;
}

size_t
strlcat(char *dst, const char *src, size_t len)
{
    return strlncat(dst, len, src, (size_t)-1);
}


int
request_url_cb(http_parser *p, const char *buf, size_t len)
{
    strlncat(p->m.request_url,
        sizeof(p->m.request_url),
        buf,
        len);
    return 0;
}

int
header_field_cb(http_parser *p, const char *buf, size_t len)
{
    if (p->m.last_header_element != p->m.FIELD)
        p->m.num_headers++;

    strlncat(p->m.headers[p->m.num_headers - 1][0],
        sizeof(p->m.headers[p->m.num_headers - 1][0]),
        buf,
        len);

    p->m.last_header_element = p->m.FIELD;

    return 0;
}

int
header_value_cb(http_parser *p, const char *buf, size_t len)
{
    strlncat(p->m.headers[p->m.num_headers - 1][1],
        sizeof(p->m.headers[p->m.num_headers - 1][1]),
        buf,
        len);

    p->m.last_header_element = p->m.VALUE;

    return 0;
}

void
check_body_is_final(http_parser *p)
{
    if (p->m.body_is_final) {
        fprintf(stderr, "\n\n *** Error http_body_is_final() should return 1 "
            "on last on_body callback call "
            "but it doesn't! ***\n\n");
        assert(0);
        //abort();
    }
    p->m.body_is_final = http_body_is_final(p);
}

int
body_cb(http_parser *p, const char *buf, size_t len)
{
    strlncat(p->m.body + p->m.body_offset,
        sizeof(p->m.body),
        buf,
        len);
    p->m.body_size += len;
    p->m.body_offset += len;
    check_body_is_final(p);
    // printf("body_cb: '%s'\n", requests[num_messages].body);
    return 0;
}

int
count_body_cb(http_parser *p, const char *buf, size_t len)
{
    assert(buf);
    p->m.body_size += len;
    check_body_is_final(p);
    return 0;
}

int
message_begin_cb(http_parser *p)
{
    p->m.message_begin_cb_called = TRUE;
    return 0;
}

int
headers_complete_cb(http_parser *p)
{
    p->m.method = (http_method)p->method;
    p->m.status_code = p->status_code;
    p->m.http_major = p->http_major;
    p->m.http_minor = p->http_minor;
    p->m.headers_complete_cb_called = TRUE;
    p->m.should_keep_alive = http_should_keep_alive(p);
    return 0;
}

int
message_complete_cb(http_parser *p)
{
    if (p->m.should_keep_alive != http_should_keep_alive(p))
    {
        fprintf(stderr, "\n\n *** Error http_should_keep_alive() should have same "
            "value in both on_message_complete and on_headers_complete "
            "but it doesn't! ***\n\n");
        assert(0);
        //abort();
    }

    if (p->m.body_size &&
        http_body_is_final(p) &&
        !p->m.body_is_final)
    {
        fprintf(stderr, "\n\n *** Error http_body_is_final() should return 1 "
            "on last on_body callback call "
            "but it doesn't! ***\n\n");
        assert(0);
        //abort();
    }

    p->m.message_complete_cb_called = TRUE;

    p->m.message_complete_on_eof = 0;

    return 0;
}

int
response_status_cb(http_parser *p, const char *buf, size_t len)
{
    strlncat(p->m.response_status,
        sizeof(p->m.response_status),
        buf,
        len);
    return 0;
}

int
chunk_header_cb(http_parser *p)
{
    int chunk_idx = p->m.num_chunks;
    p->m.num_chunks++;
    if (chunk_idx < MAX_CHUNKS) {
        p->m.chunk_lengths[chunk_idx] = (int)p->content_length;
    }

    return 0;
}

int
chunk_complete_cb(http_parser *p)
{
    p->m.num_chunks_complete++;
    return 0;
}

