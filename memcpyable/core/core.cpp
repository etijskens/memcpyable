/*
 *  C++ source file for module memcpyable.core
 */


// See http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html for examples on how to use pybind11.
// The example below is modified after http://people.duke.edu/~ccc14/cspy/18G_C++_Python_pybind11.html#More-on-working-with-numpy-arrays
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iostream>
#include <type_traits>
#include <vector>
#include <list>

#include <Eigen/Geometry>

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec_t;
typedef int64_t Index_t;

namespace mpi
{//-------------------------------------------------------------------------------------------------
    template<typename T>
    struct fixed_size_memcpy_able : std::is_trivially_copyable<T> {};
 //-------------------------------------------------------------------------------------------------
 // specializations:
    template<typename T>
    struct fixed_size_memcpy_able<T[]> : std::false_type {};

    template<typename T>
    struct fixed_size_memcpy_able<T*> : std::false_type {};

    template<typename T, int N>
    struct fixed_size_memcpy_able<Eigen::Matrix<T,N,1,Eigen::DontAlign>> : std::true_type {};
    // Also provide specializations for Vector, Point, Quaternion, ...

 //-------------------------------------------------------------------------------------------------
    template<typename T>
    struct variable_size_memcpy_able : std::false_type {};
 //-------------------------------------------------------------------------------------------------
 // specializations:
    template<typename T>
    struct variable_size_memcpy_able<std::vector<T>> : fixed_size_memcpy_able<T> {};

    template<>
    struct variable_size_memcpy_able<std::string> : std::true_type {};

 //-------------------------------------------------------------------------------------------------
    template<typename T>
    struct memcpy_traits
 //-------------------------------------------------------------------------------------------------
    {// C++17 needed
        static void* ptr(T& t)
        { 
            if constexpr(fixed_size_memcpy_able<T>::value)
                return &t; 
            else if constexpr(variable_size_memcpy_able<T>::value)
                return &t[0]; 
            else
                static_assert(fixed_size_memcpy_able<T>::value || variable_size_memcpy_able<T>::value, "type T is not memcpy-able");
        }

     // compute the size that T will occupy in the message. 
        static size_t messageSize(T& t) 
        {
            if constexpr(fixed_size_memcpy_able<T>::value) 
            {// the size of a single T
                return sizeof(T);
            }
            else if constexpr(variable_size_memcpy_able<T>::value)
            {// the size of siz_t + the size of a single T::value_type times the number of items in the collection
                return sizeof(size_t) + sizeof(typename T::value_type) * t.size();
            }
            else
                static_assert(fixed_size_memcpy_able<T>::value || variable_size_memcpy_able<T>::value, "type T is not memcpy-able");
        }

     // write a T to a buffer
        static void write(T&t, void*& dst) 
        {
            if constexpr(fixed_size_memcpy_able<T>::value)
            {// write the variable t
                memcpy( dst, ptr(t), sizeof(t) );
             // advance the pointer in the buffer
                dst = (Index_t*)(dst) + sizeof(t);
            }
            else if constexpr(variable_size_memcpy_able<T>::value)
            {// write the size of the collection:
                size_t size = t.size();
                memcpy( dst, &size, sizeof(size_t) );
             // advance the pointer in the buffer
                dst = (Index_t*)(dst) + sizeof(size_t);
             // write the collection:
                memcpy( dst, &t[0], size * sizeof(T) );
             // advance the pointer in the buffer
                dst = (Index_t*)(dst) + size * sizeof(size_t);
            }
            else
                static_assert(fixed_size_memcpy_able<T>::value || variable_size_memcpy_able<T>::value, "type T is not memcpy-able");
        }

     // read a T from a buffer
        static void read(T&t, void*& src) 
        {
            if constexpr(fixed_size_memcpy_able<T>::value)
            {// read the variable t
                memcpy( ptr(t), src, sizeof(t) );
             // advance the pointer in the buffer
                src = (Index_t*)(src) + sizeof(t);
            }
            else if constexpr(variable_size_memcpy_able<T>::value)
            {// read the size of the collection:
                size_t size;
                memcpy( &size, src, sizeof(size_t) );
             // advance the pointer in the buffer
                src = (Index_t*)(src) + sizeof(size_t);
             // resize the collection
                t.resize(size);
             // read the collection:
                memcpy( &t[0], src, size * sizeof(T) );
             // advance the pointer in the buffer
                src = (Index_t*)(src) + size * sizeof(size_t);
            }
            else
                static_assert(fixed_size_memcpy_able<T>::value || variable_size_memcpy_able<T>::value, "type T is not memcpy-able");
        }
    };
 //-------------------------------------------------------------------------------------------------
 // Two convenience template functions that allow to replace
 //     memcpy_traits<T>::write(t,dst);
 // with
 //     write(t,dst);
 // The template parameter is inferred from the first argument.
 
    template <typename T>
    void write( T&t, void*& dst)
    {
        memcpy_traits<T>::write(t,dst);
    }
    template <typename T>
    void read( T&t, void*& src)
    {
        memcpy_traits<T>::read(t,src);   
    }
 //-------------------------------------------------------------------------------------------------
}// namespace mpi

using namespace mpi;

template <typename T>
bool
test_T
  ( std::string const& nameOfT  // string with name of the class
  , bool fixed                  // expected result for fixed_size_memcpy_able<T>
  , bool variable               // expected result for variable_size_memcpy_able<T>
  )
{
    std::cout<<"\ntest_T<"<<nameOfT<<">"<<std::endl;
    std::cout<<nameOfT<<": is_standard_layout        = "<<std::is_standard_layout   <T>::value<<std::endl;
    std::cout<<nameOfT<<": is_trivially_copyable     = "<<std::is_trivially_copyable<T>::value<<std::endl;
    std::cout<<nameOfT<<": fixed_size_memcpy_able    = "<<fixed_size_memcpy_able    <T>::value<<std::endl;
    std::cout<<nameOfT<<": variable_size_memcpy_able = "<<variable_size_memcpy_able <T>::value<<std::endl;

    bool ok = true;

    ok &= (   fixed_size_memcpy_able<T>::value == fixed   );
    if( !ok ) {
        std::cout<<"!!!\n!!! fixed_size_memcpy_able tests fails for "<<nameOfT<<"\n!!!"<<std::endl;
        return ok;
    }
    
    ok &= (variable_size_memcpy_able<T>::value == variable);
    if( !ok ) {
        std::cout<<"!!!\n!!! variable_size_memcpy_able tests fails for "<<nameOfT<<"\n!!!"<<std::endl;
        return ok;
    }
    return ok;
}

template<typename T>
bool test_std_vector_T
  ( std::string const& nameOfT  // string with name of the class
  , bool fixed                  // expected result for fixed_size_memcpy_able<T>
  , bool variable               // expected result for variable_size_memcpy_able<T>
  )
{
    std::string name = "std::vector<" + nameOfT + ">";
    return test_T<std::vector<T>>(name,fixed,variable);
}



bool test_memcpy_able()
{
    bool ok = true;

    ok &= test_T<int      >("int"      , true , false);
    ok &= test_T<float    >("float"    , true , false);
    ok &= test_T<vec_t    >("vec_t"    , true , false);
    ok &= test_T<double[3]>("double[3]", true , false);
    ok &= test_T<int[]    >("int[]"    , false, false);
    ok &= test_T<int*     >("int*"     , false, false);

    ok &= test_std_vector_T<int      >("int"      , false, true);
    ok &= test_std_vector_T<float    >("float"    , false, true);
    ok &= test_std_vector_T<vec_t    >("vec_t"    , false, true);
    ok &= test_std_vector_T<double[3]>("double[3]", false, true);
    ok &= test_std_vector_T<int[]    >("int[]"    , false, false);
    ok &= test_std_vector_T<int*     >("int*"     , false, false);

    ok &= test_T<std::string>("std::string", false, true);

    std::cout<<std::endl;
    return ok;
}


bool test_traits()
{
    bool ok = true;

    typedef std::vector<int> vint_t;
    Index_t buffer[1000];
    int i0 = 1;
    int d0 = 10;
    vec_t v0(1,2,3);
    
    vint_t a0 = {10,20,30,40};
    int    i = i0; std::cout<<"i = "<<i<<std::endl;
    double d = d0; std::cout<<"d = "<<d<<std::endl;
    vec_t  v = v0; std::cout<<"v =["<<v[0]<<','<<v[1]<<','<<v[2]<<']'<<std::endl;
    vint_t a = a0; std::cout<<"a =["<<a[0]<<','<<a[1]<<','<<a[2]<<','<<a[3]<<']'<<std::endl;
    // std::cout<<"! src  = "<<&i<<", size = "<<sizeof(i)<<std::endl;

    void* ptr = &buffer[0];
    // memcpy_traits<int   >::write(i, ptr);
    // memcpy_traits<double>::write(d, ptr);
    // memcpy_traits<vec_t >::write(v, ptr);
    // memcpy_traits<vint_t>::write(a, ptr);
    write(i, ptr);
    write(d, ptr);
    write(v, ptr);
    write(a, ptr);

    i = 0;
    ok &= (i != i0); std::cout<<"i = "<<i<<std::endl;
    d = 0;
    ok &= (d != d0); std::cout<<"d = "<<d<<std::endl;
    v = vec_t(0,0,0);
    ok &= (v != v0); std::cout<<"v =["<<v[0]<<','<<v[1]<<','<<v[2]<<']'<<std::endl;
    a.clear();  
    ok &= (a.size() == 0);
    ok &= (a != a0); 
    std::cout<<"a =[";
    for( auto i : a) {
        std::cout<<i<<',';
    }   std::cout<<']'<<std::endl;

    ptr = &buffer[0];
    // memcpy_traits<int   >::read(i, ptr); 
    read(i, ptr); 
    std::cout<<"i = "<<i<<std::endl;
    ok &= (i == i0);
    // memcpy_traits<double>::read(d, ptr);
    read(d, ptr);
    std::cout<<"d = "<<d<<std::endl;
    ok &= (d == d0);
    // memcpy_traits<vec_t >::read(v, ptr);
    read(v, ptr);
    std::cout<<"v =["<<v[0]<<','<<v[1]<<','<<v[2]<<']'<<std::endl;
    ok &= (v == v0);
    // memcpy_traits<vint_t>::read(a, ptr);
    read(a, ptr);
    std::cout<<"a =["<<a[0]<<','<<a[1]<<','<<a[2]<<','<<a[3]<<']'<<std::endl;
    ok &= (a.size() == 4);
    ok &= (a == a0);
    
    return ok;
}


PYBIND11_MODULE(core, m)
{// optional module doc-string
    m.doc() = "pybind11 core plugin"; // optional module docstring
 // list the functions you want to expose:
 // m.def("exposed_name", function_pointer, "doc-string for the exposed function");
    m.def("test_memcpy_able", &test_memcpy_able);
    m.def("test_traits", &test_traits);
}
