/*
后续有待改进：
１．增加命令行参数，自动切换日志级别
２．修改-s参数功能，决定是否不管npy格式为float64或float32，都转为float32
3.检查npparray是否释放内存
*/
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

const unsigned int TOP1_CMD = 0x8001;
const unsigned int TOPN_CMD = 0x8002;

std::map<int,std::string> error_map = {
    {-1,"dimension mismatch"},
    {-2,"Fatal Error:len of all array must be same"},
    {-3,"Fatal Error:emb_array must be a two-dimensional array"},
    {-4,""},
    {-5,"Fatal Error:Invalid command"},
    {-6,"Fatal Error:Protocol error,can not decode"},
    {-7,"Fatal Error:Only support float or double type"},
    {-1000,"Fatal Error:Unkown Error"}
};
#if 1
class lsh_server
{
private:
    /* data */
    string url_workers;
    string url_router;
    unsigned int worker_num;
    unsigned int dimension;

    void *ctx;
    void *router;
    void *workers;

    unsigned long *worker_counts;
    vector<string> label;    

    bool is_simple;
    bool init_finished;
    LoggerPtr logger;

    void* dataset;
    void *qo;
    void *table;
    template <class T> 
    vector<DenseVector<T>> *get_dataset(){
        return (vector<DenseVector<T>> *)dataset;
    }
    template <class T> 
    LSHNearestNeighborQueryPool<DenseVector<T>> *get_qo(){
        return (LSHNearestNeighborQueryPool<DenseVector<T>> *)qo;//.get();
    }
    template <class T>
    void init_hash(NpyArray  &cnpy_feature){
        DenseVector<T> p;
        T* loaded_data = cnpy_feature.data<T>();
        vector<DenseVector<T>> *pdata = new vector<DenseVector<T>>;
        for(int i=0;i<cnpy_feature.shape[0];i++){
            p.resize(cnpy_feature.shape[1]);
            for(int j=0;j<cnpy_feature.shape[1];j++){
                p[j] = loaded_data[i*cnpy_feature.shape[1]+j];
            }
            pdata->push_back(p);
        }
        dataset = (void*)pdata;
        // setting parameters and constructing the table
        LSHConstructionParameters params;
        params.dimension = (*pdata)[0].size();
        params.lsh_family = LSHFamily::CrossPolytope;
        params.l = NUM_HASH_TABLES;
        params.distance_function = DistanceFunction::EuclideanSquared;
        compute_number_of_hash_functions<DenseVector<T>>(NUM_HASH_BITS, &params);
        params.num_rotations = NUM_ROTATIONS;
        // we want to use all the available threads to set up
        params.num_setup_threads = 0;
        params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;

        unique_ptr<LSHNearestNeighborTable<DenseVector<T>>> p_table = construct_table<DenseVector<T>>(*pdata, params);
        unique_ptr<LSHNearestNeighborQueryPool<DenseVector<T>>> p_qo = p_table->construct_query_pool(params.l);

        p_qo->reset_query_statistics();
        qo = (void*)p_qo.release();
        table = (void*)p_table.release();
    }

public:
    lsh_server(string feature_file,string label_file,string port,unsigned int num=10,bool b_simple=false);
    ~lsh_server();
    void loop();
    void worker(int name, string url_worker,void *ctx);
    void stater();
};

#endif
template <class T>
void construct_point_vec(const char* buf,unsigned int shape1, unsigned int shape2, vector<DenseVector<T>> *queries){
    DenseVector<T> p;
    T *datap=(T*)buf;
    for (int i=0;i<shape1;i++){  
        p.resize(shape2);
        for(int j=0;j<shape2;j++){
            p[j]=/*htond*/(datap[j]);
        }
        queries->push_back(p);
        datap+=shape2;
    }
}
template <class T>
T get_eudistance(DenseVector<T> p1,DenseVector<T> p2){
    return sqrt((p1-p2).squaredNorm());
}
template <class T>
T get_cosdistance(DenseVector<T> p1,DenseVector<T> p2){
    return p1.dot(p2)/(sqrt(p1.squaredNorm())*sqrt(p2.squaredNorm()));
}
template <class T>
T get_simi(DenseVector<T> p1,DenseVector<T> p2){
    if (get_eudistance(p1,p2)<1.0)
        return get_cosdistance(p1,p2);
    else
        return 0.0;
}
typedef std::tuple<int,vector<string>,vector<double>> ReturnTurple;
template <class T>
ReturnTurple find_neighbor(const char* buf,unsigned int shape1, unsigned int shape2,vector<float> &h_angle,vector<float> &v_angle,vector<DenseVector<T>> &dataset,vector<string> &label,unsigned int find_k,LSHNearestNeighborQueryPool<DenseVector<T>> *qo,unsigned int search_cmd,std::map<int,std::string> &error_map){    
    int error_code = 0;
    vector<string> names;
    vector<double> confidences;
    vector<DenseVector<T>> queries;
    LoggerPtr logger = Logger::getRootLogger();

    construct_point_vec(buf,shape1,shape2,&queries);
    if (queries.size() != h_angle.size() || queries.size() != v_angle.size()){
        error_code = -2;
        LOG4CXX_ERROR(logger,error_map[error_code]);
        names.push_back(error_map[error_code]);
        confidences.push_back(0.0);
        return std::make_tuple(error_code,names,confidences);
    }
    for(const auto &query : queries){
        if (search_cmd == TOP1_CMD){
            auto t1 = high_resolution_clock::now();
            int result = qo->find_nearest_neighbor(query);
            auto t2 = high_resolution_clock::now();
            double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            LOG4CXX_DEBUG(logger,"query time: " << elapsed_time);
            if (result != -1){
                names.push_back(label[result]) ;
                confidences.push_back(get_simi(query,dataset[result]));
            }
            else {
                error_code = -1000;
                LOG4CXX_ERROR(logger,error_map[error_code]);
                names.push_back(error_map[error_code]);
                confidences.push_back(0.0);
                break;
            }
        }
        else if (search_cmd == TOPN_CMD){
            auto t1 = high_resolution_clock::now();
            std::vector<int32_t> result;
            qo->find_k_nearest_neighbors(query,find_k,&result);
            auto t2 = high_resolution_clock::now();
            double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
            LOG4CXX_DEBUG(logger,"query time: " << elapsed_time);
            if (result.size() == find_k){
                for(std::vector<int32_t>::iterator itr=result.begin();itr!=result.end();itr++){
                    names.push_back(label[*itr]) ;
                    confidences.push_back(get_simi(query,dataset[*itr]));
                }
            }
            else {
                error_code = -1000;
                LOG4CXX_ERROR(logger,error_map[error_code]);
                names.push_back(error_map[error_code]);
                confidences.push_back(0.0);
                break;
            }
        }else{

        }
    }
    return std::make_tuple(error_code,names,confidences);
}


#if 1
lsh_server::lsh_server(string feature_file,string label_file,string port,unsigned int num,bool b_simple)
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

    char *loaded_label = cnpy_label.data<char>();
    char l[cnpy_label.word_size+1];
    for(int i=0;i<cnpy_label.shape[0];i++){
        memcpy(l,loaded_label+i*cnpy_label.word_size,cnpy_label.word_size);
        l[cnpy_label.word_size]='\0';
        label.push_back(l);        
    }

    dimension = cnpy_feature.shape[1];
    LOG4CXX_INFO(logger, "building the index based on the cross-polytope LSH");
    auto t1 = high_resolution_clock::now();
    is_simple = b_simple;
    if (b_simple){
        init_hash<float>(cnpy_feature);
    }
    else{
        init_hash<double>(cnpy_feature);
    }
    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    LOG4CXX_INFO(logger,"construction time: " << elapsed_time);
}

lsh_server::~lsh_server()
{
    delete [] worker_counts;
    delete dataset;
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
        for(int i=0;i<worker_num;i++){
            cur+=worker_counts[i];
        }
        LOG4CXX_INFO(logger,"lsh_server worker thread num:"<<worker_num <<" Concurrency:"<<((cur-last)/SLEEP_SECONDS));
        last=cur;
        QueryStatistics qs;
        if (is_simple){
            qs = get_qo<float>()->get_query_statistics();
        }else{
            qs = get_qo<double>()->get_query_statistics();
        }
        LOG4CXX_INFO(logger,"lsh average_total_query_time: "<<qs.average_total_query_time<<",num_queries: "<<qs.num_queries);
    }
}

void lsh_server::worker(int name, string url_worker,void *ctx)
{
    LOG4CXX_INFO(logger,"worker {" << name << "} start");
    void *worker = zmq_socket(ctx,ZMQ_REP);
    zmq_connect(worker, url_worker.c_str());
    while(1){
        int error_code = 0;
        std::tuple<int,vector<string>,vector<double>> error_return;
        vector<string> names;
        vector<double> confidences;

        /* Create an empty ØMQ message */
        zmq_msg_t msg;
        int rc = zmq_msg_init (&msg);
        assert (rc == 0);
        /* Block until a message is available to be received from socket */
        rc = zmq_msg_recv (&msg, worker, 0);
        //assert (rc != -1); 
        do {
            if (rc == -1){
                zmq_msg_close (&msg);
                error_code = -1000;
                LOG4CXX_FATAL(logger, "Fatal Error:zmq_msg_recv return -1");
                names.push_back("Fatal Error:zmq_msg_recv return -1");
                confidences.push_back(0.0);
                error_return = std::make_tuple(error_code,names,confidences);
                break;
            }
            try{
                auto t1 = high_resolution_clock::now();
                msgpack::object_handle oh = msgpack::unpack((const char*)zmq_msg_data(&msg), zmq_msg_size(&msg)); 
                auto t2 = high_resolution_clock::now();
                double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
                LOG4CXX_DEBUG(logger,"unpack msg time: " << elapsed_time);
                /* Release message */ zmq_msg_close (&msg);
                unsigned int search_cmd=0,find_k = 0;
                vector<float> h_angle;
                vector<float> v_angle;
                QueryFeature qf;
                msgpack::object  obj = oh.get();
                unsigned int moa_size = obj.via.array.size;
                if(!(obj.type == msgpack::type::ARRAY && moa_size>0)){        
                    error_code = -6;
                    LOG4CXX_ERROR(logger,error_map[error_code]);
                    names.push_back(error_map[error_code]);
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                    break;
                }
                msgpack::object *itr = msgpack::begin(obj.via.array);
                search_cmd = itr->via.i64;
                if (!(search_cmd == TOP1_CMD || search_cmd == TOPN_CMD)){
                    error_code = -5;
                    LOG4CXX_ERROR(logger,error_map[error_code]);
                    names.push_back(error_map[error_code]);
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                    break;
                }
                if (search_cmd == TOPN_CMD){
                    itr++;
                    find_k = itr->via.i64;
                }
                itr++;
                itr->convert(qf);
                itr++;
                itr->convert(h_angle);
                itr++;
                itr->convert(v_angle);

                bool is_double = false;
                if (qf.type.find('f8')!=string::npos){
                    is_double = true;
                }
                else if (qf.type.find('f4')!=string::npos){
                    is_double = false;
                }
                else{
                    error_code = -7;
                    LOG4CXX_ERROR(logger,error_map[error_code]);
                    names.push_back(error_map[error_code]);
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                    break;
                }
                if (qf.shape.size() != 2){
                    error_code = -3;
                    LOG4CXX_ERROR(logger,error_map[error_code]);
                    names.push_back(error_map[error_code]);
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                    break;
                }
                if (is_double){
                    error_return = find_neighbor<double>(qf.data.ptr,qf.shape[0],qf.shape[1],h_angle,v_angle,*get_dataset<double>(),label,find_k,get_qo<double>(),search_cmd,error_map);
                }
                else{
                    error_return = find_neighbor<float>(qf.data.ptr,qf.shape[0],qf.shape[1],h_angle,v_angle,*get_dataset<float>(),label,find_k,get_qo<float>(),search_cmd,error_map);

                }
            }
            catch(exception &e){
                if (string(e.what()).find("dimension mismatch") != std::string::npos){
                    error_code = -1;
                    LOG4CXX_ERROR(logger,string("Fatal Error:")+string(e.what()));
                    names.push_back(string("Fatal Error:")+string(e.what()));
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                }
                else{
                    error_code = -1000;
                    LOG4CXX_ERROR(logger,string("Fatal Error:")+string(e.what()));
                    names.push_back(string("Fatal Error:")+string(e.what()));
                    confidences.push_back(0.0);
                    error_return = std::make_tuple(error_code,names,confidences);
                }
            }
        }while(0);

        msgpack::sbuffer sb;
        auto t1 = high_resolution_clock::now();
        msgpack::pack(sb,error_return);
        auto t2 = high_resolution_clock::now();
        double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
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
#endif
// struct user {
//     std::string name;
//     int age;
//     std::string address;
//     MSGPACK_DEFINE_MAP(name, age, address);
// };

// struct Foo
// {
//     int i;
//     string str;
//     msgpack::type::raw_ref  data;
//     MSGPACK_DEFINE(i, str, data); 
// };


int main(int argc, char *argv[])
{
#if 1
    LoggerPtr rootLogger = Logger::getRootLogger();
    PatternLayoutPtr layout = new PatternLayout(LOG4CXX_STR("%d %level %l -%m%n"));
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
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::getDebug());

    char ch;
    bool b_simple = false;
    int temp,thread_num=10;
    string feature_file,label_file;
    string port("5556");
    while ((ch = getopt(argc,argv,"shf:l:p:t:"))!=-1)
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
        case 's':
            b_simple = true;
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

	lsh_server ls(feature_file,label_file,port,thread_num,b_simple);
    ls.loop();

    

#else
    // NpyArray cnpy_feature;
    // NpyArray cnpy_label;
    // time_t start, end;
    // time(&start);
    // //cnpy_label = cnpy::npy_load(label_file.c_str());
    // // cnpy_feature = cnpy::npy_load("./signatures.npy");
    // cnpy_label = cnpy::npy_load("./labeltest.npy");
    // cnpy_feature = cnpy::npy_load("./50w.npy");
    // time(&end);
    // std::cout << "load cost time:" << difftime(end, start) << " seconds" << std::endl;

    // printf("shape size:%d,shape1:%d,shape2:%d\n",cnpy_feature.shape.size(),cnpy_feature.shape[0],cnpy_feature.shape[1]);

    // vector<Point> feature;
    // Point point;
    // double* loaded_data = cnpy_feature.data<double>();
    // for(int i=0;i<cnpy_feature.shape[0];i++){
    //     // printf("row:%d ",i);
    //     point.resize(cnpy_feature.shape[1]);
    //     for(int j=0;j<cnpy_feature.shape[1];j++){
    //         point[j] = loaded_data[i*cnpy_feature.shape[1]+j];
    //     }
    //     feature.push_back(point);
    // }

    // printf("shape:[%d,%d]\n",feature.size(),feature[0].size());
    
    // printf("label shape:%d=%d==%d\n",cnpy_label.word_size,cnpy_label.num_vals,cnpy_label.num_bytes());

    // char *loaded_label = cnpy_label.data<char>();
    // vector<string> label;
    // char l[cnpy_label.word_size+1];
    // for(int i=0;i<cnpy_label.shape[0];i++){
    //     memcpy(l,loaded_label+i*cnpy_label.word_size,cnpy_label.word_size);
    //     l[cnpy_label.word_size]='\0';
    //     label.push_back(l);        
    // }
    // for(int i=0;i<label.size();i++)
    //     printf("----%s\n",label[i].c_str());

    // //typedef DenseVector<int> IntPoint;
    // Point p1,p2,p3,p4;
    // p1.resize(2);
    // p2.resize(2);
    // p3.resize(2);
    // p4.resize(2);
    // p1[0] = 1.0;
    // p1[1]=0;

    // p2[0] = 0;
    // p2[1]=4.0;

    // p3[0] = 0;
    // p3[1]=1.0;

    // p4[0]=-2.0;
    // p4[1]=0;

    // LOG4CXX_INFO(rootLogger,"eudistance(p1-p2):" << get_eudistance(p1,p2));
    // LOG4CXX_DEBUG(rootLogger,"eudistance(p2-p3):" << get_eudistance(p2,p3));
    // LOG4CXX_INFO(rootLogger,"cosin distance(p1-p2):" <<get_cosdistance(p1,p2));
    // LOG4CXX_INFO(rootLogger,"cosin distance(p3-p2):" <<get_cosdistance(p3,p2));
    // LOG4CXX_INFO(rootLogger,"cosin distance(p1-p4):" <<get_cosdistance(p1,p4));

    // vector<string> vs;

    // vs.push_back("hello");
    // vs.push_back("");
    // vs.push_back("end");
    // LOG4CXX_DEBUG(rootLogger, "vector string size:" << vs.size());
    // LOG4CXX_DEBUG(rootLogger, "vector string :" << vs[0]<<"|"<<vs[1]<<"|"<<vs[2]);

    //common(logger, 0);

    //BasicConfigurator::configure();
    

    // LOG4CXX_DEBUG(rootLogger, "debug message");
    // LOG4CXX_INFO(rootLogger, "info message");
    // LOG4CXX_WARN(rootLogger, "warn message");
    // LOG4CXX_ERROR(rootLogger, "error message");
    // LOG4CXX_FATAL(rootLogger, "fatal message");

    //printf("label shape:[%d,]\n",label.size());

    // std::stringstream ss;
    // user u;
    // u.name = "Takatoshi Kondo";
    // u.age = 42;
    // u.address = "Tokyo";
    // msgpack::pack(ss, u);
    // std::cout << ss.str() << std::endl;
    // msgpack::object_handle oh = msgpack::unpack(ss.str().data(), ss.str().size());
    // msgpack::object  obj = oh.get();
    // std::cout << "Unpacked msgpack object." << std::endl;
    // std::cout << obj << std::endl;
    // user u1;
    // oh.get().convert(u1);
    // std::cout << u1.name << std::endl;
    // Foo  f;
    // f.i = 4;
    // f.str = "hello world";
    // const char* tmp = "msgpack";
    // f.data.ptr = tmp;
    // f.data.size = strlen(tmp) + 1;
 
    // msgpack::sbuffer  sbuf;
    // msgpack::pack(sbuf, f);
 
    // //msgpack::unpacked  unpack;
    // msgpack::object_handle oh1 = msgpack::unpack(sbuf.data(), sbuf.size());
 
    // msgpack::object  obj1 = oh1.get();
 
    // Foo f2;
    // obj1.convert(f2);
 
    // cout << f2.i << ", " << f2.str << ", ";
    // cout << f2.data.ptr << endl;

    // SearchArg arg;
    // arg.cmd = 0x8001;
    // arg.feature.type = "<f8";
    // //arg.feature.shape = std::make_tuple(1,5);
    // arg.feature.shape.push_back(1);
    // arg.feature.shape.push_back(5);
    // //double data[][5] = {1.0,2.1,3.3,5.1,6.2};
    // // arg.feature.vec.ptr = (const char*)&data;
    // // arg.feature.vec.size = sizeof(data);
    // arg.feature.data = {{1.0,2.1,3.3,5.1,6.2}};
    // arg.h_angle = {30.2};
    // arg.v_angle = {40.1};
    // //cout << "size:" << sizeof(data) << endl;
    // msgpack::sbuffer  sbuf1;
    // msgpack::pack(sbuf1, arg);

    FILE *fp;
    char buf[20000];
    //char *buf =new(nothrow) char[600000000];
    fp = fopen("512d1.bin","rb");
    int len = fread(buf, 1, 20000, fp);
    printf("buf:%p,read size:%d\n",buf,len);
    fclose(fp);
 
    //msgpack::unpacked  unpack;
    //cout << "sbuf:" << sbuf1.data() << endl;
    //msgpack::object_handle oh2 = msgpack::unpack(sbuf1.data(), sbuf1.size());
    msgpack::object_handle oh2 = msgpack::unpack(buf, len);
 
    msgpack::object  obj2 = oh2.get();
    //cout << obj2 << endl;
    if(obj2.type == msgpack::type::ARRAY && obj2.via.array.size>0){
        cout << *obj2.via.array.ptr << endl;
        unsigned int search_cmd = msgpack::begin(obj2.via.array)->via.i64;
        cout << "search cmd:" << search_cmd << endl;
        for(msgpack::object* itr=msgpack::begin(obj2.via.array);itr!=msgpack::end(obj2.via.array);itr++){
            if (itr->type != msgpack::type::MAP)
                cout << *itr << endl;
            else{
                // for(msgpack::object_kv* itr2=msgpack::begin(itr->via.map);itr2!=msgpack::end(itr->via.map);itr2++){
                //     cout << "itr2-type:"<< itr2->val.type << endl;
                // }
                msgpack::object_kv* itr2=msgpack::begin(itr->via.map);
                msgpack::type::raw_ref  data;
                msgpack::type::raw_ref  type;
                std::vector<int> shape;
                cout << "1111:" << itr2->key.type <<", "<<itr2->val.type << endl;
                //if (string("data").compare(itr2->key)==0)
                //itr2->val.convert(data);
                itr2++;
                cout << "22222:" << itr2->key.type  <<", "<<itr2->val.type << endl;
                //itr2->val.convert(shape);
                itr2++;
                cout << "3333:" << itr2->key.type  <<", "<<itr2->val.type << endl;
                //itr2->val.convert(type);
                cout << "4444"<< endl;
                QueryFeature qf;
                itr->convert(qf);
                cout << qf.type << endl;
                cout << qf.shape[0] << ","<< qf.shape[1] << endl;
                //vector<Point> queries;
                vector<DoublePoint> double_queries;
                vector<FloatPoint> float_queries;
                void *queries;
                
                //cout << queries[0] << endl;
                bool is_double = false;
                    if (qf.type.find('f8')!=string::npos){
                        is_double = true;
                    }
                    else if (qf.type.find('f4')!=string::npos){
                        is_double = false;
                    }
                if (is_double){                                   
                    construct_point_vec<double>(qf.data.ptr,qf.shape[0],qf.shape[1],&double_queries);
                    queries = &double_queries;
                }
                else{
                    construct_point_vec<float>(qf.data.ptr,qf.shape[0],qf.shape[1],&float_queries);
                    queries = &float_queries;
                }
                do{
                    if (is_double){

                    }else{

                    }
                }while(1);
            }
        }
        // msgpack::object *feature = msgpack::begin(obj2.via.array)+1;
        // QueryFeature qf;
        // feature->convert(qf);
        // //cout << "qf.cmd:" << qf.cmd << endl;
        // cout << "quf.type:" << qf.type << endl;
    }
    // SearchArg arg2;
    // obj2.convert(arg2);
    // printf("cmd:%x\n",arg2.cmd);
    // cout << arg2.feature.type << endl;
    // printf("%d,%d\n",arg2.feature.shape[0],arg2.feature.shape[1]);
    // //cout << ((double*)arg2.feature.vec.ptr)[4] << endl;
    // printf("%f\n",arg2.feature.data[0][4]);
    // vector<Point> queries;
    // construct_point_vector(arg2.feature.data,&queries);
    // printf("query:%f\n",queries[0][4]);
    // cout << arg2.h_angle[0] << endl;
    // cout << arg2.v_angle[0] << endl;
#endif

}
