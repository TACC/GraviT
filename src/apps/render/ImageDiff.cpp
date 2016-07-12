#include <stdio.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <fstream>
using namespace std;

#include "ParseCommandLine.h"

string trim(string& str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last-first+1));
}


bool is_file_exist(const std::string fileName)
{
    std::ifstream infile(fileName.c_str());
    return infile.good();
}

unsigned int Difference(std::string fileA, std::string fileB)
{
    std::ifstream fA(fileA.c_str());
    std::ifstream fB(fileB.c_str());  

    char Ac;
    char Bc;

    unsigned int difference = 0;

    while (fA.get(Ac))          // loop getting single characters
    {
        fB.get(Bc);
        unsigned int k = (unsigned int) Ac;
        unsigned int kk = (unsigned int) Bc;
        unsigned int diff = (k > kk)? k-kk: kk-k;
        difference += diff;                                      
    }

    fA.close();
    fB.close();

    return difference;
}

int main(int argc, char **argv) 
{
    ParseCommandLine cmd("gvtImageDiff");
    cmd.addoption("diff", ParseCommandLine::PATH, "2 args, Image A & Image B", 2);
    cmd.addoption("tolerance", ParseCommandLine::INT, "The tolerance of the image.", 1);
    cmd.parse(argc, argv);

    int tolerance = 10;

    if (!cmd.isSet("diff")) {
        std::cout<<"No Images to diff."<<std::endl;
        return -1;
    }

    if (cmd.isSet("tolerance")) {
        tolerance = cmd.getValue<int>("tolerance")[0];
    }

    std::cout<<"Setting tolerance to: "<<tolerance<<std::endl;

    std::string ImageA = trim(cmd.getValue<std::string>("diff")[0]);
    std::string ImageB = trim(cmd.getValue<std::string>("diff")[1]);

    std::cout<<"Comparing: "<<ImageA<<" With : "<<ImageB<<std::endl;

    assert(is_file_exist(ImageA) == true && "File A does not exist");
    assert(is_file_exist(ImageB)  == true && "File B does not exist");

    unsigned int difference = Difference(ImageA,ImageB);

    std::cout<<"Image Difference is: "<< difference<<std::endl;

    return  difference< tolerance ? 0 : -1;
}