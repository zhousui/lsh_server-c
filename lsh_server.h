#ifndef __lsh_server_H__
#define __lsh_server_H__


#include <falconn_global.h>
using falconn::DenseVector;
using std::vector;
#ifdef f4
//typedef DenseVector<float> Point;
#else
typedef DenseVector<double> Point;
#endif
#endif