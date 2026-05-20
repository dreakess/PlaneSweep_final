#include <vector>
#include <string>

const float ZNear = 0.3f;
const float ZFar = 1.1f;
const int ZPlanes = 256;

typedef unsigned char u_char;

struct cam
{
    std::string name;
    int width;
    int height;
    int size;
    std::vector<uint8_t> Y; // Solo canale Y
    params<double> p;
    cam() : name(""), width(-1), height(-1), size(-1), Y(), p(){};
    cam(std::string _name, int _width, int _height, int _size, std::vector<uint8_t>&& _Y, params<double>&& _p)
        : name(_name), width(_width), height(_height), size(_size), Y(std::move(_Y)), p(std::move(_p)){};
};