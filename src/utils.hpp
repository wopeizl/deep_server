#pragma once
#include <string>
#include <vector>
using namespace std;

std::string encode_base64(const string& str);
std::vector<unsigned char> base64_decode(std::string const& encoded_string);
std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);

