#ifndef __msgpack_H__
#define __msgpack_H__
#include <lsh_server.h>
// #include <zmq.h>
#include <vector>

double htond(double);
namespace lshserver{
class msgpack{
    private:

    public:
    msgpack();
    ~msgpack();
    void unpack(char *,size_t,vector<Point> *);
    //void pack(void *msg);
};

}
inline lshserver::msgpack::msgpack(){

}

inline lshserver::msgpack::~msgpack(){

}
#endif