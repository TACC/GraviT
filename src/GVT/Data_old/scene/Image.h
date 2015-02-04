//
// Image.h
//


#ifndef GVT_IMAGE_H
#define GVT_IMAGE_H

#include "Color.h"

#include <string>
using namespace std;

class Image {
public:

    enum ImageFormat {
        PPM
    };

    Image(int w, int h, string fn = "mpitrace", ImageFormat f = PPM)
    : width(w), height(h), filename(fn), format(f) {
        int size = 3 * width*height;
        rgb = new unsigned char[size];
        for (int i = 0; i < size; ++i)
            rgb[i] = 0;
    }

    void Add(int pixel, float* buf) {
        int index = 3 * pixel;
        rgb[index + 0] = (unsigned char) (buf[0]*256.f);
        rgb[index + 1] = (unsigned char) (buf[1]*256.f);
        rgb[index + 2] = (unsigned char) (buf[2]*256.f);
    }

    void Add(int pixel, ColorAccumulator& ca) {
        int index = 3 * pixel;
        rgb[index + 0] = (unsigned char) (ca.rgba[0] / ca.rgba[3]*255.f);
        rgb[index + 1] = (unsigned char) (ca.rgba[1] / ca.rgba[3]*255.f);
        rgb[index + 2] = (unsigned char) (ca.rgba[2] / ca.rgba[3]*255.f);
        if (rgb[index + 0] > 255.f) rgb[index + 0] = 255;
        if (rgb[index + 1] > 255.f) rgb[index + 1] = 255;
        if (rgb[index + 2] > 255.f) rgb[index + 2] = 255;
    }

    void Add(int pixel, ColorAccumulator& ca, float w) {
        int index = 3 * pixel;
        rgb[index + 0] = ((unsigned char) (ca.rgba[0] / ca.rgba[3]*255.f) * w);
        rgb[index + 1] = ((unsigned char) (ca.rgba[1] / ca.rgba[3]*255.f) * w);
        rgb[index + 2] = ((unsigned char) (ca.rgba[2] / ca.rgba[3]*255.f) * w);
    }

    unsigned char* GetBuffer() {
        return rgb;
    }

    void Write();

    ~Image() {
        delete[] rgb;
    }

private:
    int width, height;
    string filename;
    ImageFormat format;
    unsigned char* rgb;
};


#endif // GVT_IMAGE_H
