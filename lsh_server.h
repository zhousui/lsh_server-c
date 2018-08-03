#ifndef __lsh_server_H__
#define __lsh_server_H__


#include <falconn_global.h>
#include <msgpack.hpp>
using falconn::DenseVector;
using std::vector;

typedef DenseVector<float> FloatPoint;
typedef DenseVector<double> DoublePoint;
//template <class T>
//typedef DenseVector<T> Point;
typedef DenseVector<double> Point;

struct QueryFeature {
    std::string type;
    std::vector<int> shape;
    msgpack::type::raw_ref  data;
    MSGPACK_DEFINE_MAP(type, shape, data);
};

struct SearchArg{
    unsigned int cmd;
    QueryFeature feature;
    std::vector<float> h_angle;
    std::vector<float> v_angle;
    MSGPACK_DEFINE(cmd,feature,h_angle,v_angle);
};

#endif