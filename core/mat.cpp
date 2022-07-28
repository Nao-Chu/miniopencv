//BMPImage.cpp
#include "mat.h"
#include <fstream>
#include <iostream>
#include <cstring>
 
using namespace std;
 
/**
 * condition为false时抛出错误，错误信息为error_message
 */
#define ASSERT(condition,error_message) ((condition)?0:(assertion(__FILE__, __func__, __LINE__,error_message),-1))
 
inline void assertion(const std::string &filePath, const std::string &function,
                      int line, const std::string &info){
    //获取文件名
    unsigned long long int pos=filePath.find_last_of('/');
    std::string filename(filePath.substr(pos+1));
    std::string err = filename+" "+function+" "+std::to_string(line)+">>  "+info;
    //抛出错误
    //throw std::runtime_error(err);
}
 
BMPImage::BMPImage():m_width(0),m_height(0),m_depth(8),m_row_size(0),
                     m_data(nullptr),m_color_palette_size(0),m_color_palette(nullptr) {
    ASSERT(sizeof(BMPFileHeader)==14,"The size of BMPFileHeader is not 14 bytes");
    ASSERT(sizeof(BMPInfoHeader)==40,"The size of BMPInfoHeader is not 40 bytes");
}
 
BMPImage::BMPImage(const char* bmp_file_path) {
    char buffer[500];
    sprintf(buffer,"Failed to open bmp image %s\n\0",bmp_file_path);
    ASSERT(open(bmp_file_path),buffer);
}
 
BMPImage::BMPImage(int4 width, int4 height, uint2 depth) {
    ASSERT(create(width , height, depth),"Failed to open bmp image\n");
}
 
BMPImage::~BMPImage() {
    delete[] m_data;
    delete[] m_color_palette;
}
 
bool BMPImage::open(const char* bmp_file_path) {
    clear();
    ifstream im_file(bmp_file_path,ios::binary);
    // 检查文件是否打开
    if(!im_file.is_open()){
        printf("Failed to open file %s\n",bmp_file_path);
        return false;
    }
    // -------------------读取BMPFileHeader-------------------------
    im_file.read((char *)(&m_file_header), sizeof(BMPFileHeader));
    // 判断是否是bmp图像
    if (m_file_header.bfType != 0x4d42) // 0x4d42 = 'BM'
    {
        printf("File %s is not bmp file\n",bmp_file_path);
        return false;
    }
    // --------------------读取BMPInfoHeader------------------------
    im_file.read((char *)(&m_info_header), sizeof(BMPInfoHeader));
    // ---------判断是否有调色板,如果有，则读入调色板数据-----------
    if (m_file_header.bfOffBits == 54){//说明没有调色板
        m_color_palette = nullptr;
    } else{
        // 计算调色板数量
        m_color_palette_size = (m_file_header.bfOffBits - 54)/4;
        m_color_palette = new ColorPalette[m_color_palette_size];
        // 读取调色板数据
        im_file.read((char*)(m_color_palette), m_color_palette_size * sizeof(ColorPalette));
    }
    // 给属性赋值
    m_width = m_info_header.biWidth;
    m_height = m_info_header.biHeight;
    m_depth = m_info_header.biBitCount;
    // 计算每行的字节数
    m_row_size = 4ul * (m_info_header.biBitCount * m_info_header.biWidth/32ul);
    // ---------------------------读取图像数据--------------------------------
    // 申请内存空间并同时初始化为0
    m_data = new uchar[m_info_header.biSizeImage]();
    // 读取图像数据到内存
    im_file.read((char *)m_data,m_info_header.biSizeImage);
    // 关闭文件
    im_file.close();
    return true;
}
 
bool BMPImage::save(const char* save_path) {
    if (strlen(save_path) == 0){
        printf("Your file path is empty");
        return false;
    }
    ofstream out_image(save_path,ios::binary);
    if (!out_image.is_open()){
        printf("Failed to save image %s",save_path);
        return false;
    }
    // ---------------------保存BMP文件头-----------------------
    out_image.write((char*)&m_file_header, sizeof(BMPFileHeader));
    // ---------------------保存BMP信息头-----------------------
    out_image.write((char*)&m_info_header, sizeof(BMPInfoHeader));
    // ---------------------保存图像数据------------------------
    out_image.write((char*)m_data, m_info_header.biSizeImage);
    out_image.close();
    return true;
}
 
bool BMPImage::create(int4 width, int4 height, uint2 depth) {
    // 目前不支持调色板
    clear();
    ulong bytes_per_pixel = 0;
    switch (depth){
        case 24:
            bytes_per_pixel = 3;
            break;
        case 32:
            bytes_per_pixel = 4;
            break;
        default:
            ASSERT(false,"The depth must be 24 or 32");
    }
    // 目前不支持调色板
    m_width = width;
    m_height = height;
    m_depth = depth;
    // 计算每行的字节数，每行的字节数需要是4的倍数
    m_row_size = 4ul * ((bytes_per_pixel * m_width+4ul-1ul)/4ul);
    // ---------------------初始化BMP文件头---------------------
    m_file_header.bfType = 0x4d42;
    m_file_header.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    m_file_header.bfSize = m_file_header.bfOffBits + m_height * m_row_size;
    m_file_header.bfReserved1 = 0;
    m_file_header.bfReserved2 = 0;
    // ---------------------初始化BMP信息头----------------------
    m_info_header.biSize = sizeof(BMPInfoHeader);//Should be 40
    m_info_header.biWidth = m_width;
    m_info_header.biHeight = m_height;
    m_info_header.biPlanes = 1;
    m_info_header.biBitCount = m_depth;
    m_info_header.biCompression = 0;
    m_info_header.biSizeImage = m_height * m_row_size;
    m_info_header.biXPelsPerMeter = 3780;
    m_info_header.biYPelsPerMeter = 3780;
    m_info_header.biClrUsed = 0;
    m_info_header.biClrImportant = 0;
    // 申请内存并初始化为0
    m_data = new uchar[m_info_header.biSizeImage]();
    return true;
}
 
BGR & BMPImage::rgb(uint row, uint col) {
    if(row >= m_height){
        clear();
        printf("Parameter row is big than picture height\n");
        ASSERT(false, "Parameter row is big than picture height");
    }
    if(col >= m_width){
        clear();
        printf("Parameter col is big than picture width\n");
        ASSERT(false, "Parameter col is big than picture width");
    }
    if(m_depth != 24){
        clear();
        printf("Error: Not a 24bit image\n");
        ASSERT(false, "Not a 24bit image");
    }
    BGR *rgb = nullptr;
    size_t elem_size = 3; //24位图像的的每个像素位占三个字节,24位
    // 计算索引
    size_t index = 0;
    if (m_height > 0){
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的最后一部分
        index = (m_height-row-1) * m_row_size + col*elem_size;
    }else{
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的第一部分
        index = row * m_row_size + col * elem_size;
    }
    rgb = (BGR*)(m_data+index);
    return *rgb;
}
 
BGRA &BMPImage::rgba(uint row, uint col) {
    if(row >= m_height){
        clear();
        printf("Parameter row is big than picture height\n");
        ASSERT(false, "Parameter row is big than picture height");
    }
    if(col >= m_width){
        clear();
        printf("Parameter col is big than picture width\n");
        ASSERT(false, "Parameter col is big than picture width");
    }
    if(m_depth != 32){
        clear();
        printf("Error: Not a 32bit image\n");
        ASSERT(false, "Not a 32bit image");
    }
    BGRA *rgba = nullptr;
    size_t elem_size = 4; //24位图像的的每个像素位占4个字节,32位
    // 计算索引
    size_t index = 0;
    if (m_height > 0){
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的最后一部分
        index = (m_height-row-1) * m_row_size + col*elem_size;
    }else{
        // 图像数据的存储顺序是从左下到右上，图片上的第一行存在m_data的第一部分
        index = row * m_row_size + col * elem_size;
    }
    rgba = (BGRA*)(m_data+index);
    return *rgba;
}
 
bool BMPImage::clear() {
    delete [] m_data;
    delete [] m_color_palette;
    return true;
}
 
int4 BMPImage::width() {
    return abs(m_width);
}
 
int4 BMPImage::height() {
    return abs(m_height);
}
 
uint2 BMPImage::depth() {
    return m_depth;
}