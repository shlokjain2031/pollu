//
// Created by Shlok Jain on 30/08/25.
//

#ifndef UTIL_H
#define UTIL_H
#include <vector>

/**
 * Splits a tag into a vector of strings.
 * @param  tag_value  tag to split
 * @param  delim      delimiter
 * @return the vector of strings
 */
std::vector<std::string> GetTagTokens(const std::string& tag_value, const std::string& delim_str);

/**
 * Remove double quotes.
 * @param  s
 * @return string string with no quotes.
 */
std::string remove_double_quotes(const std::string& s);

#endif //UTIL_H
