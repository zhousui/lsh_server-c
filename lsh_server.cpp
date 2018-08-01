#include <assert.h>
#include <iostream>
#include <zmq.h>
#include "cnpy.h"
#include <complex>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <memory.h>
#include <lsh_nn_table.h>
#include <core/cosine_distance.h>
#include <core/euclidean_distance.h>
#include <lsh_server.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cstdio>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include "math.h"

#include "./msgpack.h"
#include <msgpack.hpp>
#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/rolling/fixedwindowrollingpolicy.h>
#include <log4cxx/rolling/sizebasedtriggeringpolicy.h>
#include <log4cxx/filter/levelrangefilter.h>
#include <log4cxx/helpers/pool.h>
#include <log4cxx/logger.h>
#include <log4cxx/propertyconfigurator.h>
#include <log4cxx/rolling/rollingfileappender.h>
#include <log4cxx/helpers/stringhelper.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/helpers/exception.h>
#include <log4cxx/helpers/fileoutputstream.h>

//using namespace lshserver;
using namespace std;
using namespace cnpy;
using namespace log4cxx;
using namespace log4cxx::xml;
using namespace log4cxx::filter;
using namespace log4cxx::helpers;
using namespace log4cxx::rolling;

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborQueryPool;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;
using falconn::core::EuclideanDistanceDense;
using falconn::core::CosineDistanceDense;

const int NUM_PROBES = 1000;
const int NUM_HASH_TABLES = 30;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 1;
const int SLEEP_SECONDS = 5;

//LoggerPtr rootLogger; 

double get_eudistance(Point p1,Point p2){
    return sqrt((p1-p2).squaredNorm());
}

double get_cosdistance(Point p1,Point p2){
    return p1.dot(p2)/(sqrt(p1.squaredNorm())*sqrt(p2.squaredNorm()));
}

double get_simi(Point p1,Point p2){
    if (get_eudistance(p1,p2)<1.0)
        return get_cosdistance(p1,p2);
    else
        return 0.0;
}
 
class lsh_server
{
private:
    /* data */
    string url_workers;
    string url_router;
    unsigned int worker_num;

    void *ctx;
    void *router;
    void *workers;

    unsigned long *worker_counts;

    vector<Point> dataset;
    vector<string> label;
    unique_ptr<LSHNearestNeighborQueryPool<Point>> qo;
    //unique_ptr<LSHNearestNeighborQuery<Point>> qo;
    unique_ptr<LSHNearestNeighborTable<Point>> table;
    LoggerPtr logger;

public:
    lsh_server(string feature_file,string label_file,string port,unsigned int num=10);
    ~lsh_server();
    void loop();
    void worker(int name, string url_worker,void *ctx);
    void stater();
};

lsh_server::lsh_server(string feature_file,string label_file,string port,unsigned int num)
{
    url_workers = "inproc://ping-workers";
    url_router = string("tcp://*:")+port ;
    worker_num = num;
    worker_counts = new unsigned long[num];
    for(int i=0;i<num;i++)
        worker_counts[i] = 0;

    logger = Logger::getRootLogger();
    LOG4CXX_INFO(logger,"lsh_server start!!!");
    LOG4CXX_INFO(logger,"feature file:" << feature_file << ",label file:"<<label_file<<",port:"<<port<<",thread num:"<<num);

    ctx = zmq_init(1);
	assert(ctx);

    int rc;
    router = zmq_socket(ctx,ZMQ_ROUTER);
	assert(router);
    rc = zmq_bind(router,url_router.c_str());
	assert(0 == rc);

    workers = zmq_socket(ctx,ZMQ_DEALER);
    assert(workers);
    rc = zmq_bind(workers,url_workers.c_str());
	assert(0 == rc);

    time_t start, end;
    LOG4CXX_INFO(logger,"start load data");
    time(&start);

    NpyArray cnpy_feature;
    NpyArray cnpy_label;
    cnpy_label = cnpy::npy_load(label_file.c_str());
    cnpy_feature = cnpy::npy_load(feature_file.c_str());
    time(&end);
    LOG4CXX_INFO(logger, "load cost time:" << difftime(end, start) << " seconds");

    Point p;
    double* loaded_data = cnpy_feature.data<double>();
    for(int i=0;i<cnpy_feature.shape[0];i++){
        p.resize(cnpy_feature.shape[1]);
        for(int j=0;j<cnpy_feature.shape[1];j++){
            p[j] = loaded_data[i*cnpy_feature.shape[1]+j];
        }
        dataset.push_back(p);
    }
    char *loaded_label = cnpy_label.data<char>();
    //vector<string> label;
    char l[cnpy_label.word_size+1];
    for(int i=0;i<cnpy_label.shape[0];i++){
        memcpy(l,loaded_label+i*cnpy_label.word_size,cnpy_label.word_size);
        l[cnpy_label.word_size]='\0';
        label.push_back(l);        
    }

    // setting parameters and constructing the table
    LSHConstructionParameters params;
    params.dimension = dataset[0].size();
    params.lsh_family = LSHFamily::CrossPolytope;
    params.l = NUM_HASH_TABLES;
    params.distance_function = DistanceFunction::EuclideanSquared;
    compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    params.num_rotations = NUM_ROTATIONS;
    // we want to use all the available threads to set up
    params.num_setup_threads = 0;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;

    // LSHConstructionParameters params
    //     = get_default_parameters<Point>(dataset.size(),
    //                                dataset[0].size(),
    //                                DistanceFunction::EuclideanSquared,
    //                                true);
    LOG4CXX_INFO(logger, "building the index based on the cross-polytope LSH");
    auto t1 = high_resolution_clock::now();
    table = construct_table<Point>(dataset, params);
    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    //cout << "done" << endl;
    LOG4CXX_INFO(logger,"construction time: " << elapsed_time);

    qo = table->construct_query_pool(params.l);
    // qo = table->construct_query_object(params.l);
    qo->reset_query_statistics();
}

lsh_server::~lsh_server()
{
    delete [] worker_counts;
}

void lsh_server::loop()
{
    std::thread *tt = new std::thread[worker_num];
    LOG4CXX_INFO(logger,"work thread num:" << worker_num);

    time_t start, end;
    time(&start);
    //Lauch parts-1 threads
    for (int i = 0; i < worker_num; ++i) {
        tt[i] = std::thread([=] {worker(i, url_workers, ctx);});
    }
    std::thread stat_thread = std::thread([=] {stater();});
    time(&end);
    LOG4CXX_INFO(logger,"start threads consume " << difftime(end, start) << " seconds");

    zmq_device(ZMQ_QUEUE,router, workers);

    //Join parts-1 threads
    for (int i = 0; i < worker_num; ++i)
        tt[i].join();
    stat_thread.join();
    delete tt; 

    zmq_close(router);
	zmq_close(workers);
	zmq_term(ctx);

}

void lsh_server::stater(){
    unsigned long long last=0,cur=0;
    for(int i=0;i<worker_num;i++)
        last += worker_counts[i];
    while (1){
        sleep(SLEEP_SECONDS);
        cur=0;
        // printf("\n[");
        for(int i=0;i<worker_num;i++){
            cur+=worker_counts[i];
            // printf("%ld ",worker_counts[i]);
        }
        // printf("]\n");
        LOG4CXX_INFO(logger,"lsh_server worker thread num:"<<worker_num <<" Concurrency:"<<((cur-last)/SLEEP_SECONDS));
        last=cur;
        QueryStatistics qs = qo->get_query_statistics();
        LOG4CXX_INFO(logger,"lsh average_total_query_time: "<<qs.average_total_query_time<<",num_queries: "<<qs.num_queries);
    }
}

void lsh_server::worker(int name, string url_worker,void *ctx)
{
    LOG4CXX_INFO(logger,"worker {" << name << "} start");
    lshserver::msgpack mp;
    void *worker = zmq_socket(ctx,ZMQ_REP);
    zmq_connect(worker, url_worker.c_str());
    while(1){
        vector<Point> queries;
        /* Create an empty ØMQ message */
        zmq_msg_t msg;
        int rc = zmq_msg_init (&msg);
        assert (rc == 0);
        /* Block until a message is available to be received from socket */
        rc = zmq_msg_recv (&msg, worker, 0);
        assert (rc != -1); 
        //printf("%s\n",(char*)zmq_msg_data(&msg));
        auto t1 = high_resolution_clock::now();
        mp.unpack((char*)zmq_msg_data(&msg),zmq_msg_size(&msg),&queries);
        auto t2 = high_resolution_clock::now();
        double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
        //cout << "done" << endl;
        LOG4CXX_DEBUG(logger,"unpack msg time: " << elapsed_time);
        /* Release message */ zmq_msg_close (&msg);

        // for(int i=0;i<queries.size();i++){
        //     printf("[");
        //     for (int j=0;j<queries[i].size();j++)
        //         printf("%f,",queries[i][j]);
        //     printf("]\n");
        // }   

        // printf("210306[");
        //     for (int j=0;j<dataset[210306].size();j++)
        //         printf("%f,",dataset[210306][j]);
        //     printf("]\n");
        vector<string> names;
        vector<double> confidences;
        for(const auto &query : queries){
            //printf("start query-------------------------\n");
            t1 = high_resolution_clock::now();
            int result = qo->find_nearest_neighbor(query);
            //int result = qo->find_nearest_neighbor(dataset[210306]);
            //printf("end query-------------------------\n");
            t2 = high_resolution_clock::now();
            elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            LOG4CXX_DEBUG(logger,"query time: " << elapsed_time);
            if (result != -1){
                names.push_back(label[result]) ;
                confidences.push_back(get_simi(query,dataset[result]));
            }
            else {
                names.push_back("");
                confidences.push_back(0.0);
            }
        }
        msgpack::sbuffer sb;
        t1 = high_resolution_clock::now();
        msgpack::pack(sb,std::make_tuple(names,confidences));
        t2 = high_resolution_clock::now();
        elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
        LOG4CXX_DEBUG(logger,"msg pack time: " << elapsed_time);
        /* Create a new message */
        ///zmq_msg_t msg;
        rc = zmq_msg_init_size (&msg, sb.size());
        assert (rc == 0);
        /* Fill in message content  */
        memcpy(zmq_msg_data (&msg), sb.data(), sb.size());
        /* Send the message to the socket */
        rc = zmq_msg_send (&msg, worker, 0); assert (rc == sb.size());
        zmq_msg_close (&msg);
        worker_counts[name]+=1;
    }
    zmq_close(worker);
}

int main(int argc, char *argv[])
{
    LoggerPtr rootLogger = Logger::getRootLogger();
    PatternLayoutPtr layout = new PatternLayout(LOG4CXX_STR("%d %level %c -%m%n"));
    RollingFileAppenderPtr rfa = new RollingFileAppender();
    rfa->setName(LOG4CXX_STR("lsh_server"));
    rfa->setAppend(true);
    rfa->setLayout(layout);
    rfa->setFile(LOG4CXX_STR("log/lsh_server.log"));

    FixedWindowRollingPolicyPtr swrp = new FixedWindowRollingPolicy();
    SizeBasedTriggeringPolicyPtr sbtp = new SizeBasedTriggeringPolicy();

    sbtp->setMaxFileSize(2000*1024*1024);
    swrp->setMinIndex(0);

    swrp->setFileNamePattern(LOG4CXX_STR("log/lsh_server.log.%i"));
    Pool p;
    swrp->activateOptions(p);

    rfa->setRollingPolicy(swrp);
    rfa->setTriggeringPolicy(sbtp);
    rfa->activateOptions(p);
    rootLogger->addAppender(rfa); 
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getInfo());

    char ch;
    int temp,thread_num=10;
    string feature_file,label_file;
    string port("5556");
    while ((ch = getopt(argc,argv,"hf:l:p:t:"))!=-1)
    {
        switch(ch)
        {
        case 'f':
            feature_file = string(optarg);
            break;
        case 'l':
            label_file = string(optarg);
            break;
        case 'p':
            port = string(optarg);
            break;
        case 't':
            temp = atoi(optarg);
            if (temp != 0)
                thread_num = temp;
            break;
        case ':':
            printf("argment error!\n");
            break;
        case 'h':
            printf("lsh_server -f feature_file -l label_file\n");
            break;
        default:
            printf("lsh_server -f feature_file -l label_file\n");
            break;
        }
    }


	//初始化
#if 1
	lsh_server ls(feature_file,label_file,port,thread_num);
    ls.loop();
#else
    NpyArray cnpy_feature;
    NpyArray cnpy_label;
    time_t start, end;
    time(&start);
    //cnpy_label = cnpy::npy_load(label_file.c_str());
    // cnpy_feature = cnpy::npy_load("./signatures.npy");
    cnpy_label = cnpy::npy_load("./labeltest.npy");
    cnpy_feature = cnpy::npy_load("./50w.npy");
    time(&end);
    std::cout << "load cost time:" << difftime(end, start) << " seconds" << std::endl;

    printf("shape size:%d,shape1:%d,shape2:%d\n",cnpy_feature.shape.size(),cnpy_feature.shape[0],cnpy_feature.shape[1]);

    vector<Point> feature;
    Point point;
    double* loaded_data = cnpy_feature.data<double>();
    for(int i=0;i<cnpy_feature.shape[0];i++){
        // printf("row:%d ",i);
        point.resize(cnpy_feature.shape[1]);
        for(int j=0;j<cnpy_feature.shape[1];j++){
            point[j] = loaded_data[i*cnpy_feature.shape[1]+j];
        }
        feature.push_back(point);
    }

    printf("shape:[%d,%d]\n",feature.size(),feature[0].size());
    
    printf("label shape:%d=%d==%d\n",cnpy_label.word_size,cnpy_label.num_vals,cnpy_label.num_bytes());

    char *loaded_label = cnpy_label.data<char>();
    vector<string> label;
    char l[cnpy_label.word_size+1];
    for(int i=0;i<cnpy_label.shape[0];i++){
        memcpy(l,loaded_label+i*cnpy_label.word_size,cnpy_label.word_size);
        l[cnpy_label.word_size]='\0';
        label.push_back(l);        
    }
    for(int i=0;i<label.size();i++)
        printf("----%s\n",label[i].c_str());

    //typedef DenseVector<int> IntPoint;
    Point p1,p2,p3,p4;
    p1.resize(2);
    p2.resize(2);
    p3.resize(2);
    p4.resize(2);
    p1[0] = 1.0;
    p1[1]=0;

    p2[0] = 0;
    p2[1]=4.0;

    p3[0] = 0;
    p3[1]=1.0;

    p4[0]=-2.0;
    p4[1]=0;

    LOG4CXX_INFO(rootLogger,"eudistance(p1-p2):" << get_eudistance(p1,p2));
    LOG4CXX_DEBUG(rootLogger,"eudistance(p2-p3):" << get_eudistance(p2,p3));
    LOG4CXX_INFO(rootLogger,"cosin distance(p1-p2):" <<get_cosdistance(p1,p2));
    LOG4CXX_INFO(rootLogger,"cosin distance(p3-p2):" <<get_cosdistance(p3,p2));
    LOG4CXX_INFO(rootLogger,"cosin distance(p1-p4):" <<get_cosdistance(p1,p4));

    vector<string> vs;

    vs.push_back("hello");
    vs.push_back("");
    vs.push_back("end");
    LOG4CXX_DEBUG(rootLogger, "vector string size:" << vs.size());
    LOG4CXX_DEBUG(rootLogger, "vector string :" << vs[0]<<"|"<<vs[1]<<"|"<<vs[2]);

    //common(logger, 0);

    //BasicConfigurator::configure();
    

    // LOG4CXX_DEBUG(rootLogger, "debug message");
    // LOG4CXX_INFO(rootLogger, "info message");
    // LOG4CXX_WARN(rootLogger, "warn message");
    // LOG4CXX_ERROR(rootLogger, "error message");
    // LOG4CXX_FATAL(rootLogger, "fatal message");

    //printf("label shape:[%d,]\n",label.size());
#endif

}
