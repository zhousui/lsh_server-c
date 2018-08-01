// #include <zmq.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <map>
#include <arpa/inet.h> 
#include "./msgpack.h"
#include <msgpack.hpp>
#include <stdlib.h>

using namespace std;

using std::string;
using std::map;
using std::cout;
using std::endl; 
// 

long get_item_bytesize(char *buf,char **nextbuf);
unsigned int getshape(char *buf,char **nextbuf);
int bufsearch(char *src, int slen, char *flag, int flen);

void lshserver::msgpack::unpack(char *buf,size_t len, vector<Point> *queries){
    //parse shape
    //\\xa2nd\\xc3\\xa5shape
    //
    char shape_flag[]={static_cast<char>(0xa2),'n','d',static_cast<char>(0xc3),static_cast<char>(0xa5),'s','h','a','p','e',static_cast<char>(0x92),'\0'};

    int index = bufsearch(buf,len,shape_flag,strlen(shape_flag));
    // printf ("find index:%d\n",index);

    int shape1,shape2;
    char *nextb;
    char *sbuf=buf+index;
    shape1 = getshape(sbuf,&nextb);
    sbuf = nextb;
    shape2 = getshape(sbuf,&nextb);
    // printf("shape1:%d,shape2:%d,\n",shape1,shape2);

    sbuf=nextb;
    index=bufsearch(sbuf,20,"data",strlen("data"));
    sbuf=sbuf+index;
    int item_size = get_item_bytesize(sbuf,&nextb);
    // if (item_size != -1){
    //     printf("item byte size:%ld\n",item_size);
    // }
    assert(item_size != -1);

    unsigned int single_item_size;
    #ifdef f4
    single_item_size = sizeof(float);
    assert(item_size == shape1*shape2*single_item_size);
    float *datap=(float*)nextb;
    #else
    single_item_size = sizeof(double);
    assert(item_size == shape1*shape2*single_item_size);
    double *datap=(double*)nextb;
    #endif
    Point p;
    //cout << single_item_size << "   " << shape1*shape2*single_item_size << endl;
    // vector<Point> queries;
    for (int i=0;i<shape1;i++){  
        p.resize(shape2);
        //cout << i << endl;
        for(int j=0;j<shape2;j++){
            p[j]=/*htond*/(datap[j]);
        }
        queries->push_back(p);
        datap+=shape2;
    }
    //return shape1;
}

long get_item_bytesize(char *buf,char **nextbuf){
    long size = -1;

    if (*buf == static_cast<char>(0xdb)){
        size=htonl(*(uint32_t*)(buf+1));
        if (nextbuf){
            *nextbuf=(buf+5);
        }
    }//else if (l < 65536) 
    else if ((char)(*buf) == static_cast<char>(0xda)){
        size=htons(*(uint16_t*)(buf+1));
        if (nextbuf){
            *nextbuf=buf+3;
        }
    }//else if (x->use_bin_type && l < 256)
    else if ((char)(*buf) == static_cast<char>(0xd9)){
        size=*((uint8_t*)(buf+1));
        if (nextbuf){
            *nextbuf=buf+2;
        }
    }
    //if (l < 32)
    else if ((char)(*buf) & static_cast<char>(0xa0) == static_cast<char>(0xa0)){
        size=(char)(*buf) & static_cast<char>(0x0f);
        if (nextbuf){
            *nextbuf=buf+1;
        }
    }
    return size;
}

unsigned int getshape(char *buf,char **nextbuf) {    
    unsigned int shape = 0;

    if (*buf == static_cast<char>(0xce)){
        shape=htonl(*(uint32_t*)(buf+1));
        if (nextbuf){
            *nextbuf=(buf+5);
        }
    }else if ((char)(*buf) == static_cast<char>(0xcd)){
        shape=htons(*(uint16_t*)(buf+1));
        if (nextbuf){
            *nextbuf=buf+3;
        }
    }else if ((char)(*buf) == static_cast<char>(0xcc)){
        shape=*((uint8_t*)(buf+1));
        if (nextbuf){
            *nextbuf=buf+2;
        }
    }
    else{
        shape=*buf;
        if (nextbuf){
            *nextbuf=buf+1;
        }
    }
    return shape;
}

int bufsearch(char *src, int slen, char *flag, int flen)
{
    char *bp;
    char *sp = src;
    char *fp=flag;
    bp=sp;
    int index=-1;
    do{
        if (*bp==*fp){
            if (fp == flag+flen-1){
                if (bp == src+slen-1)
                    break;
                else{
                    index = bp-src+1;
                    break;
                }
            }
            bp++;
            fp++;
        }
        else{
            if (bp == src+slen-1){
                break;
            }
            sp++;
            bp=sp;
            fp=flag;
        }
    }while(1);
    return index;
}

double htond(double cursor){
    double big_endian;

    int x = 0;
    char *little_pointer = (char*)&cursor;
    char *big_pointer = (char*)&big_endian;

    while( x < 8 ){
        big_pointer[x] = little_pointer[7 - x];
        ++x;
    }

    return big_endian;

}

int __main(int,char **)
{
    char shape_flag[]={static_cast<char>(0xa2),'n','d',static_cast<char>(0xc3),static_cast<char>(0xa5),'s','h','a','p','e',static_cast<char>(0x92),'\0'};
    
    FILE *fp;
    char buf[20000];
    //char *buf =new(nothrow) char[600000000];
    fp = fopen("output.bin","rb");
    int len = fread(buf, 1, 600000000, fp);
    printf("buf:%p,read size:%d\n",buf,len);
    fclose(fp);


    int index=-1;
    index = bufsearch("ab dfffafe",strlen("ab dfffafe"),"af",strlen("af"));
    printf ("find index:%d\n",index);
    
    index = bufsearch(buf,len,shape_flag,strlen(shape_flag));
    printf ("find index:%d\n",index);

    int shape1,shape2;
    char *nextb;
    char *sbuf=buf+index;
    shape1 = getshape(sbuf,&nextb);
    sbuf = nextb;
    shape2 = getshape(sbuf,&nextb);
    printf("shape1:%d,shape2:%d,\n",shape1,shape2);

    sbuf=nextb;
    index=bufsearch(sbuf,20,"data",strlen("data"));
    sbuf=sbuf+index;
    int item_size = get_item_bytesize(sbuf,&nextb);
    if (item_size != -1){
        printf("item byte size:%ld\n",item_size);
    }

    unsigned int single_item_size;
    #ifdef f4
    single_item_size = sizeof(float);
    assert(item_size == shape1*shape2*single_item_size);
    float *datap=(float*)nextb;
    #else
    single_item_size = sizeof(double);
    assert(item_size == shape1*shape2*single_item_size);
    double *datap=(double*)nextb;
    #endif
    Point p;
    //cout << single_item_size << "   " << shape1*shape2*single_item_size << endl;
    vector<Point> queries;
    for (int i=0;i<shape1;i++){  
        p.resize(shape2);
        //cout << i << endl;
        for(int j=0;j<shape2;j++){
            p[j]=/*htond*/(datap[j]);
        }
        queries.push_back(p);
        datap+=shape2;
    }

    for(int i=0;i<queries.size();i++){
        printf("[");
        for (int j=0;j<queries[i].size();j++)
            printf("%f,",queries[i][j]);
        printf("]\n");
    }

    vector<string> vs;
    vector<double> vd;
    vs.push_back("name1");
    vs.push_back("");
    vs.push_back("na2");
    vd.push_back(0.1);
    vd.push_back(0.0);
    vd.push_back(0.85);

    msgpack::sbuffer sb;
    msgpack::pack(sb,std::make_tuple(vs,vd));

    printf("msgpack size:%d, buf:[",sb.size());
    for (int i=0;i<sb.size();i++){
        printf("%c",static_cast<char>(sb.data()[i]));
    }
    printf("]\n");

    string n1("124"),n2("a12m");
    int r1=-1,r2=-1;
    r1=atoi(n1.c_str());
    r2=atoi(n2.c_str());
    printf ("r1:%d,r2:%d\n",r1,r2);
    //delete[] buf;
    return 0;
}