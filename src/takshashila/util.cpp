#include <regex>
#include <string>
#include <vector>
//
// Created by Shlok Jain on 30/08/25.
//
namespace pollu {
    namespace takshashila {

        std::vector<std::string> GetTagTokens(const std::string& tag_value, const std::string& delim_str) {
            std::regex regex_str(delim_str);
            std::vector<std::string> tokens(std::sregex_token_iterator(tag_value.begin(), tag_value.end(),
                                                                       regex_str, -1),
                                            std::sregex_token_iterator());
            return tokens;
        }

        // remove double quotes.
        std::string remove_double_quotes(const std::string& s) {
            std::string ret;
            for (auto c : s) {
                if (c != '"') {
                    ret += c;
                }
            }
            return ret;
        }

        std::string to_lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            return s;
        }
    } // namespace takshashila
} // namespace pollu