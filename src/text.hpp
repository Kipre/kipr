/* Code borrowed from https://rosettacode.org/wiki/UTF-8_encode_and_decode#C */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "case_folding.hpp"
#include "langid_map.hpp"


typedef struct {
	char mask;    /* char data will be bitwise AND with this */
	char lead;    /* start bytes of current char in utf-8 encoded character */
	uint32_t beg; /* beginning of codepoint range */
	uint32_t end; /* end of codepoint range */
	int bits_stored; /* the number of bits from the codepoint that fits in char */
} utf_t;

utf_t pre[] = {
	{0b00111111, -128      , 0,       0,        6    },
	{0b01111111, 0         , 0000,    0177,     7    },
	{0b00011111, -64       , 0200,    03777,    5    },
	{0b00001111, -32       , 04000,   0177777,  4    },
	{0b00000111, -16       , 0200000, 04177777, 3    },
	{0},
};

utf_t * utf[] = {
	&pre[0],
	&pre[1],
	&pre[2],
	&pre[3],
	&pre[4],
	&pre[5]
};

/* All lengths are in bytes */
int codepoint_len(const uint32_t cp); /* len of associated utf-8 char */
int utf8_len(const char ch);          /* len of utf-8 encoded char */
 
char *to_utf8(const uint32_t cp);
int to_cp(const char chr[4], uint32_t * codepoint);
 
int codepoint_len(const uint32_t cp)
{
	int len = 0;
	for(utf_t **u = utf; *u; ++u) {
		if((cp >= (*u)->beg) && (cp <= (*u)->end)) {
			break;
		}
		++len;
	}
	if(len > 4) /* Out of bounds */
		std::abort();
 
	return len;
}
 
int utf8_len(const char ch)
{
	int len = 0;
	for(utf_t **u = utf; *u; ++u) {
		if((ch & ~(*u)->mask) == (*u)->lead) {
			break;
		}
		++len;
	}
	if(len > 4) { /* Malformed leading byte */
		std::abort();
	}
	return len;
}
 
char *to_utf8(const uint32_t cp)
{
	static char ret[5];
	const int bytes = codepoint_len(cp);
 
	int shift = utf[0]->bits_stored * (bytes - 1);
	ret[0] = (cp >> shift & utf[bytes]->mask) | utf[bytes]->lead;
	shift -= utf[0]->bits_stored;
	for(int i = 1; i < bytes; ++i) {
		ret[i] = (cp >> shift & utf[0]->mask) | utf[0]->lead;
		shift -= utf[0]->bits_stored;
	}
	ret[bytes] = '\0';
	return ret;
}
 
int to_cp(const char chr[4], uint32_t *code_point)
{
	int bytes = utf8_len(*chr);
	int shift = utf[0]->bits_stored * (bytes - 1);
	*code_point = (*chr++ & utf[bytes]->mask) << shift;
 
	for(int i = 1; i < bytes; ++i, ++chr) {
		shift -= utf[0]->bits_stored;
		*code_point |= ((char)*chr & utf[0]->mask) << shift;
	}
 
	return bytes;
}

/* My code */

size_t char_vectorizer(const char * text, 
	                 size_t length, 
	                 size_t nb_chars,
	                 uint16_t * result, 
	                 std::unordered_map<uint32_t, uint16_t>& charmap) {
	uint32_t code_point;
	size_t count = 0, place = 0;
	std::unordered_map<uint32_t, uint16_t>::const_iterator got;

	uint16_t whitespace = charmap[0x20], last_token = 0;

	while (count < nb_chars && place < length) {
		place += to_cp(&text[place], &code_point);
		got = charmap.find(code_point);
		if (got != charmap.end() && !(last_token == whitespace && got->second == whitespace)) {
			result[count++] = got->second;
			last_token = got->second;
		}
	}
	
	while (count < nb_chars)
		result[count++] = whitespace;

	return count;
}

void char_vectorizer(const char * text,
	                 size_t length,
	                 std::vector<uint16_t> & result,
	                 std::unordered_map<uint32_t, uint16_t> & charmap) {
	uint32_t code_point;
	size_t place = 0;
	std::unordered_map<uint32_t, uint16_t>::const_iterator got;

	uint16_t whitespace = charmap[0x20], last_token = 0;

	while (place < length) {
		place += to_cp(&text[place], &code_point);
		got = charmap.find(code_point);
		if (got != charmap.end() && !(last_token == whitespace && got->second == whitespace)) {
			result.push_back(got->second);
			last_token = got->second;
		}
	}
}

std::unordered_map<uint32_t, size_t>
count_relevant_chars(const char * text, size_t length) {
	std::unordered_map<uint32_t, size_t> countmap;
	uint32_t code_point;
	std::unordered_map<uint32_t, uint32_t>::const_iterator got;
	for(size_t i = 0; i < length;) {
		i += to_cp(&text[i], &code_point);
		got = case_fold.find(code_point);
		if (got != case_fold.end()) {
			code_point = got->second;
		}
		countmap[code_point] += 1;
	}
	return countmap;
}


std::unordered_map<uint32_t, size_t>
count_all_chars(const char * text, size_t length) {
	std::unordered_map<uint32_t, size_t> countmap;
	uint32_t code_point;
	for(size_t i = 0; i < length;) {
		i += to_cp(&text[i], &code_point);
		countmap[code_point] += 1;
	}
	return countmap;
}



void dict_to_map(std::unordered_map<uint32_t, uint16_t> & cmap, PyObject * dict) {
	PyObject * map = PyDict_Items(dict);
	Py_ssize_t len  = PySequence_Fast_GET_SIZE(map);
	PyObject ** items = PySequence_Fast_ITEMS(map);
	uint32_t key;
	uint16_t value;
	for (int i = 0; i < len; ++i) {
		PyArg_ParseTuple(items[i], "IH", &key, &value);
		cmap[key] = value;
	}
	Py_DECREF(map);
}
