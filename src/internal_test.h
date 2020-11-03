//
// microtest inspired from:
//
// URL: https://github.com/torpedro/microtest.h
//
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

////////////////
// Assertions //
////////////////

#define ASSERT(cond)\
  ASSERT_TRUE(cond);

#define ASSERT_TRUE(cond)\
  if (!(cond))  K_TEST_FAILED(#cond, __FILE__, __LINE__);

#define ASSERT_FALSE(cond)\
  if (cond) K_TEST_FAILED(#cond, __FILE__, __LINE__);

#define ASSERT_NULL(value)\
  ASSERT_TRUE(value == NULL);

#define ASSERT_NULL_AND_ERROR(value)\
  if (!(value == NULL && PyErr_Occurred())) {\
  	K_TEST_FAILED(#value, __FILE__, __LINE__);\
  }\
  PyErr_Clear();\

#define ASSERT_ERROR(value)\
  value;\
  if (!(PyErr_Occurred())) {\
    K_TEST_FAILED("Error not raised by " #value, __FILE__, __LINE__);\
  }\
  PyErr_Clear();\

#define ASSERT_NO_ERROR(value) \
value ;\
if (PyErr_Occurred()) {\
  PyErr_Print();PyErr_Format(PyExc_AssertionError, \
"Error occured for %s at line %i.", #value, __LINE__);\
  return;\
}

// #define ASSERT_NOTNULL(value)\
//   ASSERT_TRUE(value != NULL);

// #define ASSERT_STREQ(a, b)\
//   if (std::string(a).compare(std::string(b)) != 0) {\
//     printf("%s{    info} %s", mt::yellow(), mt::def());\
//     std::cout << "Actual values: " << a << " != " << b << std::endl;\
//     throw mt::AssertFailedException(#a " == " #b, __FILE__, __LINE__);\
//   }

// #define ASSERT_STRNEQ(a, b)\
//   if (std::string(a).compare(std::string(b)) !== 0) {\
//     printf("%s{    info} %s", mt::yellow(), mt::def());\
//     std::cout << "Actual values: " << a << " == " << b << std::endl;\
//     throw mt::AssertFailedException(#a " != " #b, __FILE__, __LINE__);\
//   }

#define ASSERT_CARR_EQ(a, b, len) \
  for (int i = 0; i < len; ++i) {\
      if (a[i] != b[i]) {\
        printf("%s{    info} %s", mt::yellow(), mt::def());\
        std::cout << "C array equality assertion failed, actual values: " \
        << a[i] << " != " << b[i] << " at position " << i << std::endl;\
        K_TEST_FAILED( #a " == " #b, __FILE__, __LINE__);\
      }\
  }

#define ASSERT_SHAPE_EQ(a, b) \
  for (int i = 0; i < MAX_NDIMS; ++i) {\
      if (a->shape[i] != b->shape[i]) {\
        printf("%s{    info} %s", mt::yellow(), mt::def());\
        std::cout << "Shape equality assertion failed, actual values: " << std::endl;\
        DEBUG_shape(a->shape, "Left hand: " #a);\
        DEBUG_shape(b->shape, "Right hand: "#b);\
        K_TEST_FAILED( #a "->shape == " #b "->shape", __FILE__, __LINE__);\
      }\
  }

#define ASSERT_KARR_EQ(a, b) \
  ASSERT_SHAPE_EQ(a, b) \
for (int i = 0; i < Karray_length(a); ++i) {\
  if (a->data[i] != b->data[i]) {\
    printf("%s{    info} %s", mt::yellow(), mt::def());\
    std::cout << "Karray equality assertion failed, actual values: " \
    << a->data[i] << " != " << b->data[i] << " at position " << i << std::endl;\
    K_TEST_FAILED( #a "->data == " #b "->data", __FILE__, __LINE__);\
  }\
}


#define ASSERT_EQ(a, b)\
  if (a != b) {\
    printf("%s{    info} %s", mt::yellow(), mt::def());\
    std::cout << "Actual values: " << a << " != " << b << std::endl;\
  }\
  ASSERT(a == b);

// #define ASSERT_NEQ(a, b)\
//   if (a == b) {\
//     printf("%s{    info} %s", mt::yellow(), mt::def());\
//     std::cout << "Actual values: " << a << " == " << b << std::endl;\
//   }\
//   ASSERT(a != b);


////////////////
// Unit Tests //
////////////////

#define TEST(name) \
  std::function<void (void)> name;\
  bool __##name = mt::TestsManager::AddTest(&name, #name);\
  name = [&]()


///////////////
// Framework //
///////////////

#define K_TEST_FAILED(condstr, file, line) {\
    PyErr_Format(PyExc_AssertionError, \
"Assertion failed for %s at line %i.", condstr, line); \
return; }

#define __KTEST_SETUP_ERR\
    { PyErr_SetString(PyExc_SystemError, "Could not make colored output.");\
      Py_RETURN_FALSE; }


#if defined _WIN32
#define SETUP_KTEST \
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);\
    if (hOut == INVALID_HANDLE_VALUE) __KTEST_SETUP_ERR;\
    DWORD dwMode = 0;\
    if (!GetConsoleMode(hOut, &dwMode)) __KTEST_SETUP_ERR;\
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;\
    if (!SetConsoleMode(hOut, dwMode)) __KTEST_SETUP_ERR;
#else /*__WIN32__*/
#define SETUP_KTEST
#endif /*__WIN32__*/

#define RUN_KTEST \
    size_t num_failed = mt::TestsManager::RunAllTests(stdout);\
    if (num_failed == 0) {\
      fprintf(stdout, "%s{ summary} All tests succeeded!%s\n", mt::green(), mt::def());\
      Py_RETURN_TRUE;\
    } else {\
      double percentage = 100.0 * num_failed / mt::TestsManager::tests().size();\
      fprintf(stderr, "%s{ summary} %lu tests failed (%.2f%%)%s\n", mt::red(), num_failed, percentage, mt::def());\
      Py_RETURN_FALSE;\
    };

namespace mt {

inline const char* red() {
	return "\x1b[1;31m";
}

inline const char* green() {
	return "\x1b[0;32m";
}

inline const char* yellow() {
	return "\x1b[0;33m";
}

inline const char* def() {
	return "\x1b[0m";
}

inline void printRunning(const char* message, FILE* file = stdout) {
	fprintf(file, "%s{ running}%s %s\n", green(), def(), message);
}

inline void printOk(const char* message, FILE* file = stdout) {
	fprintf(file, "%s{      ok}%s %s\n", green(), def(), message);
}

inline void printFailed(const char* message, FILE* file = stdout) {
	fprintf(file, "%s{  failed} %s%s\n", red(), message, def());
}

class TestsManager {
	// Note: static initialization fiasco
	// http://www.parashift.com/c++-faq-lite/static-init-order.html
	// http://www.parashift.com/c++-faq-lite/static-init-order-on-first-use.html
public:
	struct Test {
		const char* name;
		std::function<void (void)> * fn;
	};

	static std::vector<Test>& tests() {
		static std::vector<Test> tests_;
		return tests_;
	}

	// Adds a new test to the current set of tests.
	// Returns false if a test with the same name already exists.
	inline static bool AddTest(std::function<void (void)> * fn, const char* name) {
		tests().push_back({ name, fn });
		return true;
	}

	// Run all tests that are registered.
	// Returns the number of tests that failed.
	inline static size_t RunAllTests(FILE* file = stdout) {
		size_t num_failed = 0;

		for (const Test& test : tests()) {
			// Run the test.
			// If an AsserFailedException is thrown, the test has failed.
			printRunning(test.name, file);

			(*test.fn)();
			if (PyErr_Occurred()) {
				PyErr_Print();
				++num_failed;
				printFailed(test.name, file);
			} else {
				printOk(test.name, file);
			}

		}

		int return_code = (num_failed > 0) ? 1 : 0;
		return return_code;
	}
};
};
